# Engineering Copilot

Four internal engineering tools in one local web app. FastAPI backend, plain
HTML + vanilla JS frontend, no build step. The only outbound call is to the
Groq API on its free tier.

| Tool | Paste in | Get back |
| --- | --- | --- |
| **Unit Test Generator** | a function, class, or module | framework choice, an edge-case inventory, a runnable test file, and an honest list of what is hard to test |
| **API Doc Generator** | route/handler code | an endpoint table, valid OpenAPI 3.1 YAML, a per-endpoint reference with `curl` examples |
| **Log / RCA Summarizer** | a log excerpt | incident brief, evidence-linked timeline, root cause **with a confidence rating and contradicting evidence** |
| **Postmortem Drafter** | an incident chat transcript | a blameless postmortem with role placeholders instead of names, and specific action items |

Every tool has a **Load sample** button with a realistic input, so you can see
what it does without finding your own material first.

---

## Quick start

```bash
git clone <this repo> && cd Engineeringcopilottesttry

cp .env.example .env
# paste a free key from https://console.groq.com/keys into .env

./run.sh
```

Then open <http://127.0.0.1:8000>.

`run.sh` creates `.venv`, installs dependencies, and starts uvicorn with
reload. If you would rather drive it yourself:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

The API key is free and needs no credit card. Nothing else in this project
costs anything: no database, no cloud services, no paid packages.

---

## Read this before you set `GROQ_MODEL`

The default model for this kind of app used to be `llama-3.3-70b-versatile`.
**It no longer works on Groq's free tier.** Groq deprecated it on 2026-06-17
and decommissioned it on 2026-08-16; requests now come back
`404 model_not_found`.

This app therefore defaults to **`qwen/qwen3.6-27b`**, which is Groq's own
recommended replacement, on the free tier, with a ~131K context window.

| Model | Context | Notes |
| --- | --- | --- |
| `qwen/qwen3.6-27b` | ~131K | **Default.** Groq's recommended replacement. |
| `openai/gpt-oss-120b` | ~131K | Larger, bigger output budget. Also free tier. |
| `openai/gpt-oss-20b` | ~131K | Smaller and faster. |

Model IDs churn, so the app never trusts a baked-in list:

- `GET /api/models` proxies Groq's live model list for **your** key. This is
  the only answer that is actually current — use it, not this README.
- If you point `GROQ_MODEL` at a model the app knows is retired, it says so at
  startup, shows a banner in the UI, and names the replacement in the error.

Change the model without touching code:

```bash
GROQ_MODEL=openai/gpt-oss-120b ./run.sh
```

---

## Configuration

All of it is environment variables; see `.env.example`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | *(required)* | Free key from console.groq.com |
| `GROQ_MODEL` | `qwen/qwen3.6-27b` | Any model your key can call |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | Point at any OpenAI-compatible endpoint |
| `GROQ_TEMPERATURE` | `0.2` | Low on purpose — these are analysis tools, not creative ones |
| `GROQ_MAX_TOKENS` | `4096` | Raise it if postmortems get cut off |
| `GROQ_TIMEOUT_SECONDS` | `120` | Per-request timeout |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Local binding |

Because `GROQ_BASE_URL` is configurable and the wire format is
OpenAI-compatible, this also runs against Ollama, llama.cpp, or any other
local OpenAI-shaped server with no code changes.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/unit-tests` | `{"input": "...", "model"?, "temperature"?}` → markdown |
| `POST` | `/api/api-docs` | same shape |
| `POST` | `/api/log-rca` | same shape |
| `POST` | `/api/postmortem` | same shape |
| `GET` | `/api/tools` | tab metadata + samples (the UI builds itself from this) |
| `GET` | `/api/samples/{tool}` | one sample input |
| `GET` | `/api/config` | non-secret runtime config |
| `GET` | `/api/models` | live model list from Groq |
| `GET` | `/api/health` | liveness + whether the key is configured |
| `GET` | `/docs` | interactive OpenAPI docs (FastAPI built-in) |

Success:

```json
{
  "tool": "log-rca",
  "markdown": "## Summary\n...",
  "model": "qwen/qwen3.6-27b",
  "finish_reason": "stop",
  "elapsed_ms": 4120,
  "usage": {"prompt_tokens": 1180, "completion_tokens": 940, "total_tokens": 2120},
  "truncated": false
}
```

Failure — every error carries a remediation, not just a status code:

```json
{
  "error": "Model 'llama-3.3-70b-versatile' is unavailable: model_not_found",
  "hint": "'llama-3.3-70b-versatile' is retired: Decommissioned on the Groq free/developer tier on 2026-08-16. Set GROQ_MODEL=qwen/qwen3.6-27b in .env and restart."
}
```

---

## How it works

```
frontend/index.html  ──fetch()──►  POST /api/<tool>
                                      │
                          backend/prompts.py  (the actual product)
                                      │
                          backend/groq_client.py  ──HTTPS──►  Groq
                                      │
                          markdown ◄──┘
                                      │
              frontend/markdown.js → sanitized HTML → the page
```

```
backend/
  main.py          FastAPI app, static mount, validation-error formatting
  config.py        env config, model defaults, retired-model warnings
  groq_client.py   HTTP client + error translation
  prompts.py       the four system prompts
  samples.py       one realistic sample input per tool
  schemas.py       request/response models, input size cap
  routes/tools.py  the endpoints
frontend/
  index.html       single page, no framework
  app.js           tabs, run, copy/download, lazy Mermaid
  markdown.js      hand-rolled, escape-first markdown renderer
  styles.css       dark/light theme
tests/
  test_api.py         routing, validation, error translation (stubbed Groq)
  test_e2e_local.py   full HTTP path against a local OpenAI-compatible mock
  markdown.test.mjs   renderer, including the XSS cases
```

### Design decisions worth knowing

**The prompts are the product.** Everything else is plumbing. Each prompt pins
an exact output contract, names the failure mode it is guarding against, and
forces explicit uncertainty — the RCA tool has to state a confidence level and
list contradicting evidence, and "insufficient evidence" is an allowed answer.
The postmortem tool is forbidden from carrying human names out of a transcript.

**Pasted content is data, not instructions.** Input is fenced with explicit
delimiters and every system prompt states that anything inside is content to
analyse, never a directive. A log excerpt containing "ignore previous
instructions" gets analysed, not obeyed.

**The markdown renderer escapes first.** The whole document is HTML-escaped
before any markup is generated, so there is no raw-HTML passthrough for model
output to exploit. Links go through a protocol allowlist. This is covered by
tests that assert `<script>`, `onerror=`, and `javascript:` URLs stay inert.

**Errors tell you what to do.** A 404 from Groq becomes "this model is retired,
set `GROQ_MODEL=<replacement>`". A 429 explains the free-tier limits and
suggests another model to get a separate rate-limit bucket. A missing key gives
you the two commands that fix it.

**Mermaid loads lazily.** It is imported from the CDN only when a diagram
actually appears, and falls back to showing the diagram source if the CDN is
unreachable — so the app works fully offline apart from the Groq call itself.

**No persistence, on purpose.** No database, no accounts, no session state. The
only thing kept is in-browser: your draft and last result per tab, so switching
tabs does not lose work. Refreshing clears everything.

---

## Tests

```bash
.venv/bin/pytest              # 39 backend tests
node --test tests/markdown.test.mjs   # 22 renderer tests
```

No network, no API key, and no cost: `test_api.py` stubs the Groq call,
`test_e2e_local.py` stands up a local server that speaks Groq's wire protocol
and drives the whole stack over real HTTP.

If your shell has an HTTP proxy configured, exclude loopback so the
local-mock tests can connect:

```bash
NO_PROXY=127.0.0.1,localhost .venv/bin/pytest
```

---

## What is not built yet

The stack was specified with two things these four tools do not need, so they
are deliberately not wired up:

- **Zip upload handling** (`zipfile`). No current tool takes a repo archive;
  all four take pasted text. Nothing here imports `zipfile`.
- **Mermaid diagrams** are rendered when a model emits a ` ```mermaid ` block
  (the RCA prompt's timelines are the natural fit), but no tool is *required*
  to produce one yet.

Both are in place to build on rather than pretended to be finished.
