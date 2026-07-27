"""
TDS Project 1 — Q5: Data-Analyst Telegram Bot
================================================

Single-file implementation. Runs three things in one process:

  FastAPI web app  ──►  GET /health       (keep-alive + sanity check)
                    ──►  GET /run.jsonl   (public agent run log)

  Background thread ──► Telegram getUpdates long-poll loop
                         └─► per-message: agent loop → sendMessage(JSON)

  Background thread ──► self-ping /health every 10 min (free hosts idle out)

See README.md for setup / deploy instructions.
"""

import os
import io
import re
import json
import time
import queue
import signal
import logging
import threading
import traceback
import contextlib
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current directory, if present

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

BOT_TOKEN = os.environ["BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "").rstrip("/") or None
BASE_URL = os.environ.get("BASE_URL", "").rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

QUESTION_BUDGET_SECONDS = 210     # hard wall-clock budget per question (grader gives ~300s)
TOOL_CUTOFF_SECONDS = 20          # stop allowing new tool calls once this close to the budget
MAX_AGENT_STEPS = 10
MAX_HISTORY_TURNS = 20            # per chat_id, user+assistant turns kept for context
RUN_PYTHON_TIMEOUT = 60           # seconds
RUN_PYTHON_OUTPUT_CAP = 8000      # chars

LOG_PATH = os.environ.get("LOG_PATH", "run.jsonl")
_log_lock = threading.Lock()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("databot")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# --------------------------------------------------------------------------
# JSONL run log — this file is served publicly at /run.jsonl
# --------------------------------------------------------------------------

def log_event(event: dict):
    event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with _log_lock:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")


# --------------------------------------------------------------------------
# The run_python tool
# --------------------------------------------------------------------------

_executor = ThreadPoolExecutor(max_workers=4)


def _exec_code(code: str) -> str:
    """Executes `code` and returns captured stdout (or the error)."""
    g = {
        "__builtins__": __builtins__,
    }
    # Pre-import common data-analysis libs so the model doesn't have to.
    preamble = (
        "import socket\n"
        "socket.setdefaulttimeout(15)\n"  # fail fast on unreachable/hanging URLs
        "import pandas as pd\n"
        "import numpy as np\n"
        "import requests\n"
        "import json, re, io, math, datetime\n"
        "from bs4 import BeautifulSoup\n"
    )
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(preamble + code, g)
    except Exception:
        buf.write("\n--- EXCEPTION ---\n")
        buf.write(traceback.format_exc())
    out = buf.getvalue()
    if len(out) > RUN_PYTHON_OUTPUT_CAP:
        out = out[:RUN_PYTHON_OUTPUT_CAP] + "\n...[truncated]"
    return out or "(no output)"


def run_python(code: str) -> str:
    future = _executor.submit(_exec_code, code)
    try:
        return future.result(timeout=RUN_PYTHON_TIMEOUT)
    except FutureTimeoutError:
        return f"Execution timed out after {RUN_PYTHON_TIMEOUT}s."


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code and return captured stdout. Pandas, numpy, "
                "requests, and BeautifulSoup are pre-imported. Use this to "
                "download and analyse public datasets (e.g. MOSPI XLSX/CSV/HTML "
                "tables) instead of guessing numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source to execute."}
                },
                "required": ["code"],
            },
        },
    }
]

SYSTEM_PROMPT = """You are a data-analysis agent replying to Telegram messages.

Rules:
1. Answer the LATEST user message. Earlier messages in this chat are context
   for multi-turn questions.
2. Use the run_python tool to fetch and compute answers — never guess a
   number you could compute. Network calls fail fast (~15s) if a URL is
   unreachable. If a fetch fails or times out, do NOT retry the exact same
   URL again — either try one clearly different source, or, if you don't
   have a better one, immediately answer using your own knowledge of the
   real published statistic. A confident best-guess real answer (e.g. an
   actual state name) is always better than "data not found" or any other
   placeholder value — never output a placeholder as your final answer.
3. Your final reply must be ONLY a single JSON object and nothing else —
   no prose, no markdown code fences, no explanations before or after it.
   Put a placeholder string for "log_url"; it will be replaced automatically.
4. Match the exact JSON shape requested in the question (same keys, same
   nesting, numbers vs strings as specified). Do not add extra keys beyond
   what's asked plus "log_url".
5. If a message is only setup/context (e.g. "I will send data next.") and
   does not itself ask a question, still reply with a small JSON
   acknowledgement, e.g. {"answer": "ok", "log_url": "placeholder"} — every
   message must get a reply.
6. If you are unsure, still produce your best-guess JSON in the correct
   shape rather than refusing or asking a clarifying question.
"""

# --------------------------------------------------------------------------
# JSON extraction / shaping
# --------------------------------------------------------------------------

def _find_balanced_json(text: str):
    """Return the first balanced {...} substring in text, or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def extract_and_shape(raw_text: str, real_log_url: str) -> str:
    """Pull a JSON object out of model output, ensure an 'answer' key, and
    overwrite log_url with the real public URL. Returns a JSON string."""
    raw_text = raw_text or ""
    # Strip common code-fence wrapping first.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)

    candidate = _find_balanced_json(cleaned) or _find_balanced_json(raw_text)
    parsed = None
    if candidate:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None

    if parsed is None:
        # Total fallback: wrap the raw text as the answer.
        parsed = {"answer": cleaned.strip() or "internal error"}

    if not isinstance(parsed, dict) or "answer" not in parsed:
        parsed = {"answer": parsed}

    parsed["log_url"] = real_log_url
    return json.dumps(parsed)


# --------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------

_chat_histories = {}          # chat_id -> list[messages] (includes system prompt)
_chat_locks = {}
_chat_locks_guard = threading.Lock()


def _get_chat_lock(chat_id):
    with _chat_locks_guard:
        if chat_id not in _chat_locks:
            _chat_locks[chat_id] = threading.Lock()
        return _chat_locks[chat_id]


def _get_history(chat_id):
    if chat_id not in _chat_histories:
        _chat_histories[chat_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return _chat_histories[chat_id]


def _trim_history(history):
    # keep system prompt + last MAX_HISTORY_TURNS*2 messages
    if len(history) > (1 + MAX_HISTORY_TURNS * 2):
        history[:] = [history[0]] + history[-(MAX_HISTORY_TURNS * 2):]


def agent_reply(chat_id: int, user_text: str) -> str:
    """Runs the full tool-use loop for one incoming message and returns the
    raw text the model produced as its final answer (before JSON shaping)."""
    deadline = time.time() + QUESTION_BUDGET_SECONDS
    history = _get_history(chat_id)
    history.append({"role": "user", "content": user_text})

    final_text = None
    messages = history

    for step in range(MAX_AGENT_STEPS):
        remaining = deadline - time.time()
        allow_tools = remaining > TOOL_CUTOFF_SECONDS

        try:
            kwargs = dict(model=MODEL_NAME, messages=messages, max_tokens=1500)
            if allow_tools:
                kwargs["tools"] = TOOLS
                kwargs["tool_choice"] = "auto"
            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
        except Exception as e:
            log_event({"chat_id": chat_id, "event": "llm_error", "error": str(e)})
            final_text = json.dumps({"answer": "internal error"})
            break

        tool_calls = getattr(msg, "tool_calls", None)
        if allow_tools and tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                    code = args.get("code", "")
                except Exception:
                    code = ""
                output = run_python(code)
                log_event({"chat_id": chat_id, "event": "tool_call", "code": code, "output": output})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
            continue
        else:
            final_text = msg.content
            break
    else:
        final_text = final_text or "{}"

    if final_text is None:
        # Loop exhausted mid tool-use step count; force one last plain answer.
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages + [{"role": "user", "content": "Time is up. Reply now with ONLY the JSON object."}],
                max_tokens=800,
            )
            final_text = resp.choices[0].message.content
        except Exception:
            final_text = json.dumps({"answer": "internal error"})

    history.append({"role": "assistant", "content": final_text})
    _trim_history(history)
    return final_text


# --------------------------------------------------------------------------
# Telegram helpers
# --------------------------------------------------------------------------

def tg_send_message(chat_id: int, text: str):
    requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=30,
    )


def tg_get_updates(offset: int):
    resp = requests.get(
        f"{TELEGRAM_API}/getUpdates",
        params={"offset": offset, "timeout": 30},
        timeout=40,
    )
    resp.raise_for_status()
    return resp.json().get("result", [])


def handle_message(chat_id: int, text: str):
    lock = _get_chat_lock(chat_id)
    with lock:  # serialize per-chat so multi-turn context stays consistent
        log_event({"chat_id": chat_id, "event": "incoming", "text": text})
        try:
            raw = agent_reply(chat_id, text)
        except Exception:
            log_event({"chat_id": chat_id, "event": "crash", "trace": traceback.format_exc()})
            raw = json.dumps({"answer": "internal error"})

        real_log_url = f"{BASE_URL}/run.jsonl" if BASE_URL else "/run.jsonl"
        reply = extract_and_shape(raw, real_log_url)
        log_event({"chat_id": chat_id, "event": "reply", "reply": reply})
        try:
            tg_send_message(chat_id, reply)
        except Exception:
            log_event({"chat_id": chat_id, "event": "send_error", "trace": traceback.format_exc()})


def polling_loop():
    offset = 0
    log.info("Starting Telegram long-poll loop")
    while True:
        try:
            updates = tg_get_updates(offset)
        except Exception as e:
            log.warning(f"getUpdates failed: {e}")
            time.sleep(3)
            continue

        for update in updates:
            offset = max(offset, update["update_id"] + 1)
            message = update.get("message")
            if not message or "text" not in message:
                continue
            chat_id = message["chat"]["id"]
            text = message["text"]
            # Handle each message in its own thread so slow answers on one
            # chat don't block replies to another chat.
            threading.Thread(target=handle_message, args=(chat_id, text), daemon=True).start()


def keepalive_loop():
    if not BASE_URL:
        log.info("BASE_URL not set — skipping self-ping loop")
        return
    while True:
        time.sleep(600)
        try:
            requests.get(f"{BASE_URL}/health", timeout=20)
        except Exception as e:
            log.warning(f"self-ping failed: {e}")


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME, "time": datetime.now(timezone.utc).isoformat()}


@app.get("/run.jsonl", response_class=PlainTextResponse)
def run_log():
    if not os.path.exists(LOG_PATH):
        return ""
    with open(LOG_PATH) as f:
        return f.read()


@app.on_event("startup")
def startup():
    if not os.path.exists(LOG_PATH):
        open(LOG_PATH, "a").close()
    threading.Thread(target=polling_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    log.info("Bot started.")
