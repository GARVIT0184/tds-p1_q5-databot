# TDS Project 1 — Q5: Data-Analyst Telegram Bot

An LLM agent that answers data-analysis questions over Telegram and replies
with a single JSON object, per the grading spec.

## How it works

- **FastAPI** serves `GET /health` (keep-alive) and `GET /run.jsonl` (public
  run log — this is your `log_url`).
- A background thread long-polls Telegram's `getUpdates` and dispatches each
  incoming message to the agent loop.
- The agent loop gives the model one tool, `run_python`, which executes code
  server-side (pandas/numpy/requests/BeautifulSoup pre-imported) and returns
  captured stdout. The model loops: call tool → read output → call tool again
  → ... → final answer.
- The final model output is parsed for a balanced `{...}` JSON object,
  wrapped in `{"answer": ...}` if needed, and `log_url` is overwritten with
  the real public URL before sending.
- Every tool call, every reply, and every incoming message is appended to
  `run.jsonl` (one JSON object per line).

## 1. Create the Telegram bot

1. In Telegram, message **@BotFather** → `/newbot`.
2. Pick a display name, then a **username ending in `bot`**
   (e.g. `yourname_databot`).
3. Save the HTTP API token it gives you — this is `BOT_TOKEN`.

No webhook needed; this bot uses long polling.

## 2. Configure environment variables

```bash
cp .env.example .env
```

Fill in:
- `BOT_TOKEN` — from BotFather
- `OPENAI_API_KEY` — a **direct** API key (not a proxy token that can expire
  before grading happens)
- `BASE_URL` — set this once you know your deployed URL (step 4)
- `MODEL_NAME` — defaults to `gpt-4o`. Don't downgrade to a mini model —
  small models get real-world stats questions wrong.

## 3. Run locally

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
export $(cat .env | xargs)     # or use python-dotenv / your OS's env loading
uvicorn bot:app --host 0.0.0.0 --port 8000
```

Then in another terminal:

```bash
curl http://localhost:8000/health
```

Message your bot from your own Telegram account and confirm you get back a
single clean JSON reply.

## 4. Deploy (Render free tier)

1. Push this repo to a **public GitHub repo** (required for grading).
2. On Render: **New → Web Service**, connect the repo. It picks up
   `render.yaml` automatically (or set manually):
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn bot:app --host 0.0.0.0 --port $PORT`
3. Add env vars `BOT_TOKEN`, `OPENAI_API_KEY`, `BASE_URL` in the Render
   dashboard. `BASE_URL` = `https://<your-service>.onrender.com` (no
   trailing slash).
4. **Important:** changing env vars on Render does *not* restart the
   service — trigger a manual deploy afterwards.
5. Free instances spin down after ~15 min idle. This app self-pings its own
   `/health` every 10 minutes to stay warm — but if the grader hits it cold
   after a long gap, the first reply may be slow. Consider also adding an
   external pinger (e.g. UptimeRobot) hitting `/health` every 5–10 min for
   extra insurance.

Verify:

```bash
curl https://<your-host>/health      # {"ok": true, ...}
wget https://<your-host>/run.jsonl   # must download publicly
```

## 5. Test like the grader tests

- Message your bot from your own Telegram account (a real user account —
  exactly what the grader uses). Send:

  > Which state has the highest maternal mortality rate based on MOSPI data?
  > Reply with ONLY this JSON object and nothing else:
  > {"answer": {"state": "<state name>"}, "log_url": "..."}

  Confirm you get back exactly one clean JSON object.
- Test a multi-turn flow: send `"I will send data next."` then follow up
  with data + a question. Confirm the bot replies to *both* messages.
- `wget` your `log_url` from a different network to confirm it's truly
  public and shows the run you just did.
- Optional full dress rehearsal: clone the public grading pipeline
  ([Jivraj-18/tds-p1-t2-2026-telegram-bot](https://github.com/Jivraj-18/tds-p1-t2-2026-telegram-bot)),
  point it at your bot, and add your own questions to
  `evals/questions.json`.

## 6. Register on SEEK

One box, comma-separated, in this exact order:

```
https://github.com/<you>/<your-repo>, your_bot_username
```

- Repo URL first, then the bot **username** (no `@`), which must end in
  `bot`.
- Click **Check**, then **Save**.

## Checklist before you walk away

- [ ] Bot replies to a fresh Telegram message with exactly one JSON object
- [ ] `answer` shape matches whatever the message asked for
- [ ] `log_url` in the reply is wget-able and shows the run you just did
- [ ] Multi-turn: bot replies to *every* message
- [ ] Reply always arrives well under 300s (test a hard question)
- [ ] Repo is public; no secrets committed (`.env` is git-ignored)
- [ ] Host stays awake (keep-warm ping working)
- [ ] LLM credentials will still be valid weeks from now (grading happens
      after the deadline)
- [ ] Registered on SEEK, Checked, **Saved**

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `format_error` | Prose/fences around the JSON, or two messages sent | Check `extract_and_shape`; make sure the model isn't wrapping in ```json fences (it's told not to, but the extractor also strips them defensively) |
| `timeout` | Cold-started host, slow dataset fetch with no answer budget | Confirm keep-warm ping is running; `QUESTION_BUDGET_SECONDS` forces an answer before 300s |
| Wrong answers on stats questions | Model too small | Keep `MODEL_NAME=gpt-4o` or better |
| Bot dead at grading time | Expired API token, or free host asleep | Use a direct (non-expiring) API key; verify keep-alive |
| Multi-turn question scored zero | Bot only replied to the last message | Every incoming Telegram message triggers its own `handle_message` call — confirm none are being dropped in your polling loop |
| `bad_bot` | Wrong username registered / bot never started | Double-check the SEEK registration string and that `/health` returns 200 |

## Notes on the code

- `run_python` executes in a `ThreadPoolExecutor` with a 60s timeout per
  call and an 8000-char output cap.
- Conversation history is kept per `chat_id`, capped at the last 20 turns,
  so multi-turn questions have context without the prompt growing unbounded.
- A per-chat `threading.Lock` serializes messages within the same chat (so
  multi-turn order is preserved) while different chats are handled
  concurrently.
- `QUESTION_BUDGET_SECONDS` (210s) and `TOOL_CUTOFF_SECONDS` (20s) keep
  replies well under the grader's ~300s timeout — once the budget is tight,
  tool calls are disabled and the model is forced to answer with what it
  has.
