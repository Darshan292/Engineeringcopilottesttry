# Engineering Copilot

Four internal engineering tools in one local web app. FastAPI backend, plain
HTML + vanilla JS frontend, no build step. Everything is free and open source;
the only outbound call is to the Groq API on its free tier.

The point of the design: **the LLM is one stage in a pipeline, not the pipeline.**
Input is redacted, classified and parsed before it reaches the model, and
everything the model returns is verified before it reaches you. See
[ARCHITECTURE.md](ARCHITECTURE.md) for why each stage exists.

| Tool | Paste in | Get back |
| --- | --- | --- |
| **Unit Test Generator** | a function, class or module | tests written against the boundary conditions extracted from the AST, **then actually executed** with the pass/fail result shown |
| **API Doc Generator** | route/handler code | OpenAPI 3.1 that is **schema-validated and cross-checked** against the routes and error paths found in the source |
| **Log / RCA Summarizer** | a log excerpt | incident brief where every claim cites a line ID that is **verified to exist**, with a confidence score computed in code |
| **Postmortem Drafter** | an incident chat transcript | blameless postmortem where names were **removed before the model saw anything** |

Each tool has a **Load sample** button with a realistic input.

---

## Quick start

```bash
cp .env.example .env
# paste a free key from https://console.groq.com/keys into .env

./run.sh
```

Open <http://127.0.0.1:8000>.

No credit card, no database, no cloud services. Dependencies are FastAPI,
uvicorn, httpx, pydantic, python-dotenv, PyYAML and openapi-spec-validator —
all MIT/BSD/Apache-2.0.

---

## Read this before setting `GROQ_MODEL`

The conventional default `llama-3.3-70b-versatile` **no longer works**. Groq
deprecated it on 2026-06-17 and decommissioned it on 2026-08-16; requests now
return `404 model_not_found`.

The default here is **`qwen/qwen3.6-27b`**, Groq's own recommended replacement.

| Model | Notes |
| --- | --- |
| `qwen/qwen3.6-27b` | **Default.** ~131K context, free tier. |
| `openai/gpt-oss-120b` | Larger. Follows JSON schemas more reliably — worth switching to if you see repair attempts. |
| `openai/gpt-oss-20b` | Smaller and faster. |

Model IDs churn, so the app never trusts a baked-in list. `GET /api/models`
returns the live list for your key, and pointing `GROQ_MODEL` at a model known
to be retired produces an error naming the replacement.

Because `GROQ_BASE_URL` is configurable and the wire format is
OpenAI-compatible, this also runs against Ollama, llama.cpp, vLLM or LM Studio
with no code changes. A loopback URL automatically bypasses any `HTTP_PROXY` in
your environment.

---

## What makes this more than a prompt wrapper

**Secrets never leave the machine.** AWS keys, JWTs, connection strings, GitHub
and Slack tokens, private keys, emails and card numbers are detected and
replaced with stable placeholders before the request is built, and a final gate
refuses to send if anything survived. Private IPs are deliberately kept —
`10.4.2.0/24` is the evidence in an RCA. The response tells you exactly what was
withheld, by kind, never by value.

**Structure is extracted, not inferred.** Python goes through `ast`, so the
prompt receives the literal boundary conditions (`subtotal > 10000`,
`customer_tier != 'platinum'`), the exceptions actually raised, and which calls
cross a dependency boundary. Route code yields the `raise HTTPException(409)`
paths buried in handler bodies — the errors hand-written docs always miss. Logs
are field-split and clustered into templates. Transcripts are anonymized.

**Large inputs are compressed, not truncated.** Repeated log messages collapse
into templates with counts and first/last occurrence: a 60,000-line log
compresses about 242x. When it still does not fit, the input is split, **every
part is analysed in its own call**, and a combine step reasons over the partial
findings plus the global overview — so a cause in part 2 and its effects in
part 5 remain connectable. When even that will not work, you get a refusal with
a reason and options, never a silent analysis of 3% of the log.

**A call may only cite what that call saw.** Evidence IDs are scoped per call,
so the model cannot cite a real line it was never shown — which would otherwise
sail through the grounding check because the line exists.

**The model returns JSON, not prose.** It fills a typed schema; this app renders
the Markdown. Section order and tables are identical every run, a missing field
is a validation error with a path, and a bounded repair loop (2 retries) feeds
the exact error back.

**Claims are checked against the input.** Every cited evidence ID is verified
against what the parser produced. An ID that does not exist is named as a
fabrication and any timeline row resting on it is removed and reported.

**Confidence is computed, not asked for.** Citation validity, claim coverage,
evidence relevance, input parse rate, context completeness and whether
disconfirming evidence was sought. The model's own rating is shown beside it,
and a material gap is flagged as overconfidence.

**Generated tests are executed.** In a subprocess against your actual source.
"Verified: 3 of 3 tests pass" is a result. A failure feeds the repair loop with
the real pytest output.

**Generated OpenAPI is validated.** Parsed, checked against the OpenAPI 3.1
schema, and cross-checked against the routes found in the source — so an
omitted endpoint or an invented one is caught.

**Blameless is structural.** Names are replaced before the prompt exists and the
mapping is never serialized. The model cannot leak a name it was never shown.

---

## Governance

All optional, all defaulting to sensible local-use values.

| Control | Default | Variable |
| --- | --- | --- |
| Rate limit | 20/min, 500/day, shed locally | `RATE_LIMIT_PER_MINUTE`, `RATE_LIMIT_PER_DAY` |
| Concurrency | 4 simultaneous upstream calls | `MAX_CONCURRENT_REQUESTS` |
| Auth | off (pointless on localhost) | `API_TOKENS` |
| CORS | loopback origins only | `CORS_ORIGINS` |
| Model allowlist | any | `ALLOWED_MODELS`, `ALLOW_MODEL_OVERRIDE` |
| Test execution | on; sandboxed via `bwrap` if installed | `ENABLE_TEST_EXECUTION` |
| Redaction posture | redact and continue | `REDACTION_FAIL_CLOSED`, `REDACTION_PATTERNS` |

Binding to a non-loopback interface without auth logs a security warning at
startup naming the specific risk. Every response carries a request ID, in the
body and the `X-Request-ID` header, and a per-stage timing trace.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/unit-tests` | `{"input": "...", "model"?, "temperature"?}` |
| `POST` | `/api/api-docs` | same shape |
| `POST` | `/api/log-rca` | same shape |
| `POST` | `/api/postmortem` | same shape |
| `GET` | `/api/tools` | tab metadata + samples (the UI builds itself from this) |
| `GET` | `/api/config` | non-secret config, governance state, your rate-limit usage |
| `GET` | `/api/models` | live model list from Groq |
| `GET` | `/api/health` | liveness + whether the key is configured |
| `GET` | `/docs` | interactive OpenAPI docs |

A successful response carries the rendered document plus everything needed to
audit it:

```json
{
  "tool": "log-rca",
  "request_id": "4e295766e9f1",
  "markdown": "# Incident Brief\n...",
  "warnings": ["2 secret(s) were removed before the request left this machine: ..."],
  "attempts": 1,
  "repairs": [],
  "diagnostics": {
    "redaction":  {"redacted_count": 2, "by_kind": {"AWS_ACCESS_KEY": 1}, "contained_credentials": true},
    "injection":  {"detected": true, "max_severity": "high", "kinds": ["instruction_override"]},
    "parse":      {"total_lines": 26, "parse_rate": 0.96, "distinct_templates": 16, "first_anomaly_id": "L3"},
    "grounding":  {"citations_total": 9, "citations_valid": 8, "citations_fabricated": ["L9999"]},
    "confidence": {"computed_score": 0.35, "computed_band": "low", "model_claimed_band": "high",
                   "model_overconfident": true, "factors": [...]}
  },
  "trace": {"stages": [{"name": "redact", "duration_ms": 1}, ...]}
}
```

Errors carry a remediation, not just a status code:

```json
{
  "error": "Model 'llama-3.3-70b-versatile' is unavailable: model_not_found",
  "hint": "'llama-3.3-70b-versatile' is retired: Decommissioned on the Groq free/developer tier on 2026-08-16. Set GROQ_MODEL=qwen/qwen3.6-27b in .env and restart.",
  "request_id": "4e295766e9f1"
}
```

---

## Tests

```bash
.venv/bin/pytest                        # 213 tests, no network, no key, no cost
node --test tests/markdown.test.mjs     # 22 renderer tests including XSS
```

The suites cover the deterministic layer, every parser, the validation layer
(including really executing generated tests), the API and governance, and an
adversarial set: prompt injection, leaked secrets, fabricated citations,
malformed code, 50k-line logs, contradictory evidence and repair-loop bounds.

**These prove the system behaves correctly. They say nothing about whether the
model's analysis is any good.** That needs a real model:

```bash
export GROQ_API_KEY=...
python -m evals.run                  # 5 golden cases, 31 checks, 9 critical
python -m evals.run --repeat 3       # exposes non-determinism
python -m evals.run --model openai/gpt-oss-120b --verbose
```

Checks are deterministic predicates over the document and diagnostics — no
LLM-as-judge, because grading a model's output with another model inherits the
same failure modes. Critical checks (no leaked names, no fabricated citations,
tests that execute, valid OpenAPI) fail the run regardless of the aggregate
score. `--repeat` matters: a single green run hides the cases that pass only
most of the time.

---

## Project layout

```
backend/
  core/         redaction, detection, tokens, injection, chunking, IR
  parsers/      logs (+ template mining), code (AST), routes, transcripts
  validation/   schemas, grounding, confidence, openapi, python_exec
  pipeline/     structured calls with repair, tracing, the four tool pipelines
  render/       validated structures -> Markdown
  governance.py rate limits, model policy, auth, CORS
frontend/       single page, vanilla JS, hand-rolled escape-first Markdown
tests/          209 tests, all free to run
evals/          golden set, run against a real model
```

---

## Not built yet

- **Zip upload handling.** No tool takes a repo archive; all four take pasted
  text. Nothing imports `zipfile`.
- **Mermaid diagrams.** Rendering support exists in the frontend but no tool is
  required to emit one.
- **tree-sitter parsing** for non-Python languages. They currently get
  regex-based signature extraction at ~0.45 confidence, and the prompt is told
  to qualify its claims accordingly. This is the clearest next improvement.

Stated rather than pretended to be finished. See the **Honest limits** section
of [ARCHITECTURE.md](ARCHITECTURE.md) for what the verification layer does and
does not prove.
