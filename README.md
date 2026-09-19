# Engineering Copilot

Four internal engineering tools in one local web app. FastAPI backend, plain
HTML + vanilla JS frontend, no build step. Everything is free and open source;
the only outbound call is to OpenRouter's free tier, and the app refuses to
make a call it has not first proven costs nothing.

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
# paste a free key into .env -- https://openrouter.ai/settings/keys
# no credit card required, and a $0 balance is enough for the free models

.venv/bin/python scripts/list_models.py --free   # what your key can actually call
# copy one of those ids into LLM_MODEL in .env, then:

.venv/bin/python scripts/preflight.py            # prove it works and costs nothing
./run.sh
```

`preflight.py` is worth running once. It checks the key, loads the catalogue,
confirms your model is priced at zero, then makes **one real call** and asserts
the provider billed it at nothing. It spends one request out of your daily 50;
`--no-call` skips that and checks configuration only.

It matters more than a usual smoke test here: `openrouter.ai` was unreachable
from the environment this was built in, so the live API path is the one part
that was never exercised during development. See
[Honest limits](ARCHITECTURE.md#honest-limits).

Open <http://127.0.0.1:8000>.

No credit card, no database, no cloud services. Dependencies are FastAPI,
uvicorn, httpx, pydantic, python-dotenv, PyYAML and openapi-spec-validator —
all MIT/BSD/Apache-2.0.

---

## OpenRouter, and only free models

The target is OpenRouter's free tier. Any other OpenAI-compatible endpoint also
works — Ollama, llama.cpp, vLLM, LM Studio — because the client speaks the same
wire format, but OpenRouter is the one the guards are built around.

```bash
OPENROUTER_API_KEY=sk-or-v1-...
LLM_MODEL=              # pick one -- see below
```

**Do not trust any model list written in this repository.** Free variants appear
and vanish as providers donate and withdraw capacity, so nothing is compiled in.
Ask your own key instead:

```bash
.venv/bin/python scripts/list_models.py --free
```

No default model ships, deliberately. Hundreds of models, free variants that
come and go — any ID baked in here would eventually become a confusing 404 about
a model you never chose. Being asked to pick beats being told your model is
gone.

`openrouter/free` is a valid choice: it routes only within free models. The
router `openrouter/auto` is **blocked by name**, because it selects across the
whole catalogue including paid models and no price check can see where it went.

### OpenRouter is not just a different base URL

This is the part that bites, and it is why `backend/providers.py` exists rather
than a base-URL setting:

| | Typical OpenAI-style API | OpenRouter (free) |
|---|---|---|
| Rations | **tokens** per minute | **requests**: 20/min, 50/day (1,000/day above $10 lifetime spend) |
| 429 states the delay | `retry-after` + in the body | often **nothing at all** |
| Reset header | a duration | a **Unix millisecond timestamp** |
| Model list field | `context_window` | `context_length` |

Each of those has a failure mode that is silent or absurd rather than obvious:

- Treating "no stated delay" as "give up" reproduces, on OpenRouter, exactly the
  bug the retry loop exists to fix. A 429 with nothing in it now backs off from
  a per-provider default instead.
- Reading a reset *instant* as a *duration* is a sleep of roughly fifty thousand
  years. The parser checks the value's magnitude as well as the profile's flag,
  so a stale table cannot cause it.
- Budgeting a request-rationed provider against a token ceiling caps every call
  at a limit that does not exist. Leave `TOKEN_LIMIT_PER_MINUTE` unset on
  OpenRouter; unset means "not rationed" and the full context window is used.

Request and token limits both default to the active provider's free tier, so
the local limiter — not an upstream 429 — is normally what you hit first.

---

## It cannot spend your money

`FREE_TIER_ONLY` is on by default, and it is not a naming convention or a
promise to be careful. Nothing is callable until the provider's own price list
has shown every one of its rates to be zero.

Three independent layers, because any one of them can be wrong:

| Layer | When | What it catches |
|---|---|---|
| Price list | before the call | a paid model, a model absent from the catalogue, a model quoting no prices, a `:free` suffix typo'd away, a router that selects paid models |
| `provider.max_price = 0` | in the request | a catalogue that went stale between the check and the call, and a router whose choice is made after we stop looking |
| Reported cost | after the call | anything the first two did not foresee — and it **latches**, refusing every later call until the process is restarted |

Refusals happen before a socket is opened, so a misconfiguration costs nothing
at all. `openrouter/auto` is named explicitly and blocked: it is priced at zero
and selects from the whole catalogue, so no pricing check can see the cost.
`openrouter/free` is allowed, because it routes only within free models.

**One honest caveat.** Layer 1 only works where the provider publishes prices.
OpenRouter does, per model, which is why the guard is built around it. Point
`LLM_BASE_URL` at an endpoint that publishes none — a local Ollama, or a
provider where "free tier" is a property of your account rather than of a model
— and there is nothing per-model to check, so only layer 3 applies.
`GET /api/config` reports which regime is actually in force under
`billing.mode`, rather than implying a guarantee that is not being made.

## Requests are the scarce resource, not tokens

Most OpenAI-style APIs ration tokens per minute. **OpenRouter's free tier
rations requests**: 20 a minute, and 50 a day until you have bought $10 of
credit. That changes what "expensive" means, and it is the single most important
difference to understand here.

One browser click is one HTTP request and anything from one to thirty upstream
calls. The provider counts the upstream ones. So the limiter counts them too,
and the planner budgets against what is left today — a fourteen-call plan with
twelve calls remaining is refused and degraded to a single call, with the
arithmetic shown, rather than spending the rest of your afternoon.

## Rate limits are a delay, not a failure

A free-tier allowance is small enough that a normal request hits it. The
application treats that as pacing, not as an error:

- **Before the call**, the local limiter holds the request until the minute has
  room. It refuses upfront only when a single call could never fit, and then it
  says so with the arithmetic instead of letting you retry into a wall.
- **After the call**, a `429` is read for the delay the provider states
  (`retry-after`, `x-ratelimit-reset-tokens`, or the "try again in 4.3s" in the
  message body), waited out, and retried. Up to six times, sharing one
  `MAX_TOTAL_WAIT_SECONDS` budget with the local pacing so a request never waits
  longer in total than you allowed.
- **A configured limit is a ceiling.** `TOKEN_LIMIT_PER_MINUTE=8000` stays 8,000
  even when the provider's headers advertise 70,000. Discovery may lower the
  budget; it does not overrule your decision.
- **The 429 is also the best information available.** Its body names the limit
  that actually bound, the model that actually ran, and what the provider
  counted this call as costing. All three are adopted.

### Splitting solves size, not rate limits

Worth being explicit, because it is easy to assume otherwise: map-reduce exists
for input that is **larger than one call**. It does not make a small allowance
go further — it does the opposite. Every part re-sends the system prompt and
re-reserves the output, so N parts cost roughly N times that fixed overhead.

If the per-call overhead already dominates your allowance, the app says so in a
warning rather than splitting into something slower *and* more expensive. The
fixes there are a larger allowance or a cheaper model, and the warning names
both.

### Routing models cost more than they look like

`openrouter/free` is a router: it picks a free model per request rather than
being one. Routers have consequences this app handles explicitly:

- The rate limit that binds is the **underlying** model's, so a 429 can name a
  model you never chose. The error says so rather than leaving you to wonder.
- The prompt on the wire carries routing and tool instructions you never wrote,
  so a call costs more than the visible input suggests. Routers start with that
  correction applied rather than discovering it by failing.
- `openrouter/auto` is a different thing entirely and is **refused**: it routes
  across the whole catalogue, paid models included, so no price check can see
  where your request went. `openrouter/free` stays inside free models, which is
  why it is allowed.

---

## Read this before setting `LLM_MODEL`

There is no default, and that is deliberate. OpenRouter's free catalogue turns
over constantly — variants appear when a provider donates capacity and vanish
when they stop. Any ID written into this repository would eventually 404 for
someone who never chose it.

So ask your own key what it can call:

```bash
.venv/bin/python scripts/list_models.py --free
```

That prints only models whose every pricing field is zero — the same check the
billing guard applies before each call. Copy an ID into `LLM_MODEL`.

Two practical notes:

- **Prefer a model that is reliable at structured output.** Every call here asks
  for JSON against a schema, and a model that wanders costs you repair attempts,
  which on a 50-request day is real money in requests.
- **`openrouter/free` is a reasonable first choice** if you do not want to pick:
  it routes only within free models and adapts as the catalogue changes.

### Not every model in that list is a chat model

The provider's `/models` endpoint returns everything your key can call —
including classifiers, speech-to-text and text-to-speech. Picking one of those
does not work, and it is an easy mistake because they look like normal entries.

`meta-llama/llama-prompt-guard-2-86m`, for instance, is a 512-token
prompt-injection classifier that returns a label. `GET /api/models` now marks
every entry `usable: true/false` with a reason, and selecting an unusable one
fails immediately with a named remedy instead of somewhere deep in the pipeline.

### Tokens per minute is the limit you will actually hit

Free-tier accounts are limited far more tightly by **tokens** per minute than by
requests. `openai/gpt-oss-120b` allows 30 requests/min but only **8,000
tokens/min**, and one request from this app costs several thousand — instructions,
extracted facts, and a structured reply. Two or three requests in a minute can
exhaust the allowance.

The app now tracks that budget locally and refuses over-budget requests before
they reach the provider, so a burst costs nothing upstream. It adopts the real
limits from the provider's `x-ratelimit-*` response headers where they are sent,
and falls back to `TOKEN_LIMIT_PER_MINUTE` / `TOKEN_LIMIT_PER_DAY`. Remaining
budget is shown next to the model name in the UI.

Because `LLM_BASE_URL` is configurable and the wire format is
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
part 5 remain connectable. With many parts the combine itself becomes a tree:
batches are merged, then the merges are merged, so the final call never
overflows.

**Reading everything beats reading the loudest parts.** The strategy ladder is
ordered by how much of your input survives, not by what is cheapest:

| Strategy | Calls | What is lost |
|---|---|---|
| `full` | 1 | nothing |
| `summary` | 1 | repeated lines become a pattern with counts; no line is unaccounted for |
| `map_reduce` | N+1 | nothing — every line is read by exactly one part |
| `selected` | 1 | the text of low-scoring lines; they stay counted in the pattern table |
| `reject` | 0 | — |

`selected` is the only lossy strategy, so it is the **fallback**, not the
preference. It is reached only when reading everything is genuinely out of
budget, and when that happens the plan says so, shows the arithmetic that ruled
full coverage out, and names what would buy it back. The model is told, in the
plan text it receives, that it is looking at a subset — a model that believes it
read everything states its conclusions flatly and is wrong.

**You can see the plan before you pay for it.** `POST /api/plan/{tool}` and the
**Plan** button run the same redaction, parsing, budgeting and planner the real
run uses, and make no model call at all. You get what was extracted before the
model is involved, how the input will be split, which parts carry which line
IDs, how many calls it will take and roughly how long — including a refusal and
its reason, if that is what would happen. It is the same code path as the run,
because a preview that can disagree with the run is a lie with a progress bar.

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
| Request limit | 20/min, 500/day, shed locally | `RATE_LIMIT_PER_MINUTE`, `RATE_LIMIT_PER_DAY` |
| **Token budget** | 8k/min, 200k/day. A value you set is a **ceiling**: provider headers may lower it, never raise it | `TOKEN_LIMIT_PER_MINUTE`, `TOKEN_LIMIT_PER_DAY` |
| Provider back-off | a 429 is waited out and retried, up to 6 times, using the delay the provider states | shares `MAX_TOTAL_WAIT_SECONDS` |
| Concurrency | 4 simultaneous upstream calls | `MAX_CONCURRENT_REQUESTS` |
| Pacing | wait for token budget rather than refuse | `WAIT_FOR_TOKEN_BUDGET` |
| Max wait, one request | 600s total spent waiting on the allowance | `MAX_TOTAL_WAIT_SECONDS` |
| Max planned duration | 600s; past this, full coverage degrades to selection | `MAX_PLAN_SECONDS` |
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
| `POST` | `/api/plan/{tool}` | same shape — what the run **would** do. No model call, no tokens. |
| `GET` | `/api/tools` | tab metadata + samples (the UI builds itself from this) |
| `GET` | `/api/config` | non-secret config, governance state, your rate-limit usage |
| `GET` | `/api/models` | live model list for your key, each marked free or not |
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
  "error": "Model 'some-vendor/withdrawn-model:free' is unavailable: model_not_found",
  "hint": "OpenRouter does not serve 'some-vendor/withdrawn-model:free' on this account. Open GET /api/models to see what your key can actually call, then set LLM_MODEL in .env.",
  "request_id": "4e295766e9f1"
}
```

---

## Tests

```bash
.venv/bin/pytest                        # 331 tests, no network, no key, no cost
node --test tests/markdown.test.mjs     # 22 renderer tests including XSS
```

The suites cover the deterministic layer, every parser, the validation layer
(including really executing generated tests), the API and governance, and an
adversarial set: prompt injection, leaked secrets, fabricated citations,
malformed code, 50k-line logs, contradictory evidence and repair-loop bounds.

Two suites exist because the thing they check was once wrong and the existing
tests could not see it:

- `test_statelessness.py` proves no content carries between requests, using two
  vocabulary-disjoint inputs, so leakage would be unmistakable rather than
  plausible.
- `test_plan_preview.py` proves the plan shown before a run is the plan the run
  follows — including that the promised model-call count bounds the real one on
  a split input, which is where it was previously wrong by 2x.
- `test_rate_limit_recovery.py` proves a 429 is waited out and retried rather
  than surfaced, that a configured token limit is never raised by the provider,
  and that the limit and real per-call cost stated in a 429 body are adopted.
- `test_free_tier_guard.py` proves the three billing layers, including that the
  model list never labels a model free that the guard would refuse — the two
  used to disagree for exactly the router that made it dangerous.
- `test_providers.py` proves the things OpenRouter does differently from a
  typical OpenAI-style API — in particular that its Unix-millisecond reset
  instant is never read as a duration, even if the provider profile's flag is
  wrong.

**These prove the system behaves correctly. They say nothing about whether the
model's analysis is any good.** That needs a real model:

```bash
export OPENROUTER_API_KEY=...
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
