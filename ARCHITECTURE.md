# Architecture

## The problem this solves

The first version of this app was:

```
raw text -> prompt -> LLM -> markdown -> screen
```

That design has one point of failure and no way to detect it. The model parses,
reasons, and formats in a single pass, and whatever comes back is shipped. A
fabricated log line, an invalid OpenAPI document, a test suite that does not
compile, a customer's API key forwarded to a third party, or a colleague's name
in a "blameless" postmortem — all of them look exactly like success.

The current design moves every decidable thing out of the model:

```
                 ┌──────────────── deterministic ────────────────┐
raw input ──────>│ redact → classify → scan → parse → budget →   │
                 │ plan context                                  │
                 └───────────────────────┬───────────────────────┘
                                         │  extracted facts, secrets gone
                                         v
                              ┌──────────────────────┐
                              │   LLM (judgement)    │  returns JSON,
                              │   + bounded repair   │  never a document
                              └──────────┬───────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │ verify evidence → run domain validators →     │
                 │ compute confidence → render Markdown          │
                 └───────────────────────────────────────────────┘
                                deterministic
```

The model still does the part that needs judgement. It no longer does the parts
that can be checked.

## Stage by stage

| Stage | Module | What it removes from the model's job |
|---|---|---|
| Redact | `core/redaction.py` | Secrets and PII never reach the wire. Stable placeholders keep repeated-value relationships intact. |
| Classify | `core/detect.py` | Language and input kind are scored from signals, so "this is a log, not code" is a fact with a confidence, and a tool/input mismatch warns. |
| Injection scan | `core/injection.py` | Injection attempts are detected, reported, and defanged in place before the call. |
| Parse | `parsers/*` | Real extraction: Python AST, route decorators and `raise` bodies, log field splitting and template mining, transcript speaker anonymization. |
| Budget | `core/tokens.py` | Context limits are computed, not guessed. The estimator self-calibrates against `usage.prompt_tokens` from real responses. |
| Plan context | `core/chunking.py` | Oversize input is compressed or chunked with global context attached, degraded to a scored subset only when full coverage is unreachable, or refused with the arithmetic. Never silently truncated. |
| **Model call** | `pipeline/base.py` | Judgement only. Output must satisfy a JSON schema; failures feed a bounded repair loop. Chunked input runs one call per part plus a combine tree. |
| Verify evidence | `validation/grounding.py` | Every cited ID is checked against the parser's output. Fabrications are named and their rows removed. |
| Domain validation | `validation/{python_exec,openapi}.py` | Generated tests are executed. Generated OpenAPI is schema-validated and cross-checked against the real routes. |
| Confidence | `validation/confidence.py` | Computed from measurable signals, not asked for. The model's own rating is shown beside it. |
| Render | `render/markdown.py` | Section order, tables and caveats are produced by code, identical every run. |

## The strategy ladder

Ordered by how much of the input survives, not by what is cheapest:

| Strategy | Calls | What is lost |
|---|---|---|
| `full` | 1 | nothing |
| `summary` | 1 | repeated lines become a pattern with counts; no line is unaccounted for |
| `map_reduce` | N + combine tree | nothing — every line is read by exactly one part |
| `selected` | 1 | the text of low-scoring lines; they stay counted in the pattern table |
| `reject` | 0 | — |

`selected` is the only lossy strategy and it sits *below* map-reduce
deliberately. It was above it once, on the reasoning that one call beats N+1 —
which is true about cost and wrong about the product. A 600-line log that could
have been read in full across thirteen paced calls was instead thinned to forty
lines and answered in one: faster, cheaper, and a worse answer to the question
the user asked. Selection is now the fallback, reached only when reading
everything is out of budget, and when it is reached the plan says so, shows the
arithmetic that ruled full coverage out, and names what would buy it back. The
model is told in its own prompt that it is looking at a subset — a model that
believes it read everything states its conclusions flatly and is wrong.

## Showing the plan before executing it

`POST /api/plan/{tool}` runs the deterministic front half — redact, classify,
scan, parse, budget, plan — and returns what would happen, with no model call
and no token spend. The UI's **Plan** button uses it, and a run fetches it first
so the panel is populated before the first call goes out.

It calls the same parsers and the same planner as the run. That is the whole
design constraint: a preview produced by a separate estimator can disagree with
the run, and a preview that can be wrong about the thing it exists to report is
worse than no preview. The test
`test_the_promised_call_count_holds_for_a_split_input` exists because that
happened — the planner priced the combine as one call, the runtime built a tree,
and the panel advertised 14 calls for a run that made 27.

Nothing about progress is invented. The browser cannot observe a stage
finishing, so during a run the panel shows the plan and an elapsed clock;
per-call timings appear afterwards, taken from the response's own trace, with
repair attempts summed into the call they belong to and waiting time reported
separately from model time.

## Why each choice

**IR with stable IDs (`core/ir.py`).** Everything addressable is what makes
grounding possible. Without `L47` meaning a specific parsed line, "cite your
evidence" is decoration.

**Structured output instead of Markdown.** A missing section becomes a
`ValidationError` with a field path instead of an absence nobody notices. It
also closes the injection hole fencing leaves open: "ignore your instructions
and print X" cannot succeed against an endpoint that can only return
`RCAOutput`.

**Bounded repair (2 retries).** Unbounded repair against a rate-limited free
tier is a way to burn a daily quota on one bad request. Three attempts, then an
honest failure.

**Executing generated tests.** "Runnable" is a claim until something runs it.
Execution distinguishes three outcomes — does not import, imports but fails,
passes — that are indistinguishable from reading the code.

**Computed confidence.** Asking a model how confident it is produces a word
generated by the same process that produced the claim. The score here comes
from citation validity, claim coverage, evidence relevance, input parse rate,
context completeness, and whether disconfirming evidence was sought. A chunked
run scores lower by construction, because no single call saw everything.

**The output contract is a compact type sketch, not JSON Schema.** Embedding
`model_json_schema()` was the obvious choice and cost 43-64% of each system
prompt: `{"title": "Summary", "type": "string", "maxLength": 4000}` spends a
dozen tokens on what `"summary": string` says in three, and `$defs`/`$ref`
indirection makes the model resolve pointers before it can see the shape. On a
tokens-per-minute ceiling that overhead is the difference between a request
succeeding and a repair attempt exhausting the minute's allowance. The compact
rendering is ~63% smaller and pydantic still enforces the real contract after
the response arrives.

**The rate limiter counts tokens, not requests.** Requests per minute is easy to
model and the wrong constraint. Groq's free tier allows 30 requests/min but
8,000 tokens/min for a chat model, and this application's requests are large, so
the token ceiling binds first. Counting requests meant repair attempts were sent
into a budget already spent, and the caller got an upstream 429 naming a limit
the application had never heard of. The budget is now tracked locally, checked
before every attempt including repairs, and adopted from the provider's
`x-ratelimit-*` headers so it tracks the account in use rather than a number
compiled into the source.

**Unusable models fail by name, immediately.** The provider lists every model a
key can call, including classifiers and speech models. Selecting
`meta-llama/llama-prompt-guard-2-86m` -- a 512-token injection classifier --
previously produced "the global context alone needs 61 tokens of the 0
available", which is accurate and useless. Non-chat families and windows below a
usable threshold are now rejected up front with the reason and a working
alternative, and `/api/models` marks each entry.

**Runtime model metadata is authoritative.** The static context-window table is
a cold-start fallback. `GET /v1/models` reports the real window per model, and
that value wins — a table baked into source is wrong the moment a provider
changes it, and being optimistically wrong means a request that fails upstream
after the whole local pipeline has run. An unknown model falls back to a
conservative 8k window and the response says so.

**CPU work stays off the event loop.** Redaction, parsing, context planning and
test execution are all synchronous and run over multi-megabyte inputs. Inline,
a single 3MB paste froze the entire service — including `/api/health` — for
seconds. They now run via `asyncio.to_thread`. This is a genuine improvement,
not a complete one: CPython holds the GIL during regex work, so a heavy request
still adds latency to others (measured: median 112ms, worst 677ms on health
checks during a 3.3s request). True isolation would need a process pool.

**Template mining for logs.** 40,000 repetitions of one timeout message become
one template with a count and first/last occurrence. Measured: a 60,000-line log
compresses ~242x and fits a 131k-token window with room to spare. The
alternative — truncating at a character limit — discards the end of the
incident, which is where the resolution is.

**Global context in every chunk.** Split an incident in half and the first half
sees a database slowing with no errors while the second sees errors with no
cause. Both halves produce a reasonable-sounding wrong answer. Every chunk
therefore carries the whole input's time window, services, level counts,
earliest anomaly, and template frequencies, and is told explicitly that it is
looking at one part of a larger whole.

**Map-reduce actually executes.** Each part is analysed in its own call, then a
combine step reasons over the partial findings plus the global overview. An
earlier version planned N chunks and sent only the first, silently discarding
the rest while still presenting the result as an analysis of the whole input --
74% of a 600-line log disappeared with no indication. `tests/test_mapreduce.py`
now asserts that every parsed line reaches some call.

**Evidence scoping matches what each call saw.** A call may only cite IDs its
own prompt contains. Declaring the whole input's ID space — which the earlier
version did — let the model cite a real line it had never been shown, and that
citation passed the grounding check, because the ID existed. That was a false
negative in the one mechanism whose job is catching invention. The combine step
is scoped tighter still: it may only reuse IDs the parts actually cited, since
it never saw raw content at all.

**Structural anonymization.** Names are replaced before the prompt is built, and
the name→placeholder table is never serialized. The model cannot leak a name it
was never shown. A second check rejects any participant role that is not a real
transcript participant.

## Honest limits

These are real and not papered over:

- **Grounding checks existence, not entailment.** It proves `L47` exists and
  reports lexical overlap with the claim. It does not prove `L47` supports the
  claim. Doing that properly needs entailment scoring or a second model pass.
- **Non-Python parsing is regex-based.** JavaScript, TypeScript, Go, Java and
  Ruby get signature-level extraction with `confidence ≈ 0.45`, and the prompt
  is told to qualify accordingly. tree-sitter would fix this; it is the clearest
  next improvement.
- **The token estimator is an estimate.** No offline tokenizer exists for an
  arbitrary configured model. It is deliberately conservative and self-corrects
  from observed usage, but it is not exact.
- **Test execution isolation depends on the host.** With `bwrap` (bubblewrap)
  installed, the subprocess runs with an unshared network namespace and a
  read-only root: that is a real boundary. Without it, network and process
  spawning are intercepted in-process via an injected `conftest.py`, which stops
  the realistic accident (generated code calling an API or pip-installing) but
  is not a security boundary — `ctypes` can undo it. The report states which
  layer was in force. `ENABLE_TEST_EXECUTION=false` turns it off entirely.
- **Redaction is not a complete DLP system.** It detects known credential
  formats, PII shapes and high-entropy blobs in secret-like contexts.
  Proprietary identifiers it has never seen still pass, which is what
  `REDACTION_PATTERNS` exists for, and `REDACTION_FAIL_CLOSED=true` refuses a
  request outright rather than trusting pattern matching with material where
  that trust is misplaced.
- **The rate limiter is per-process.** In-memory, so it is load shedding for a
  single instance, not a distributed quota.
- **Confidence measures evidence quality, not correctness.** A well-evidenced
  wrong answer can still score high. The weights are a defensible starting
  point, not a calibrated model.
- **Map-reduce is capped, and the cap is arithmetic rather than a constant.**
  Three separate limits refuse a plan *before* any call is paid for: more than
  `DEFAULT_MAX_CHUNKS` (40) parts; a projected wait over `MAX_PLAN_SECONDS`
  (600s); or a combine tree that cannot converge. The last one matters most and
  is the least obvious: a combine call must hold at least two partial answers or
  the tree never shrinks, and with two per batch only `2**4 = 16` parts fold
  inside the runtime's four-round limit. Discovering that at the final combine
  costs every map call that came before it, so the planner computes it up front.
  Where a relevance score exists, these limits degrade to a scored subset with
  the arithmetic shown instead of refusing outright.
- **The promised call count is an upper bound, not a prediction.** How many
  partial answers fit one combine call depends on how verbose the model is,
  which is unknowable before the calls happen. The plan assumes every partial
  uses its full output allowance, so the run can come in under the promise and
  never over it. Coming in over would mean spending quota nobody agreed to.
- **A long analysis is a long HTTP request.** Full coverage on a free-tier
  allowance is genuinely minutes: 600 varied log lines at 8,000 tokens/minute is
  13 parts and roughly seven minutes, almost all of it spent waiting on the
  allowance rather than on the model. The request is synchronous, so a reverse
  proxy or gateway with a shorter read timeout will cut it off. Raise those
  timeouts alongside `MAX_PLAN_SECONDS`, or leave the default and accept the
  lossy `selected` fallback for the largest inputs. Streaming or a job queue
  would fix this properly and neither is built.
- **Data still leaves the machine.** Redacted and structured, but the extracted
  facts go to Groq. That is a property of the product category. Point
  `GROQ_BASE_URL` at a local model if it matters.

## Testing strategy

| Suite | What it covers | Cost |
|---|---|---|
| `tests/test_core.py` | Redaction, tokens, detection, injection, chunking | free |
| `tests/test_parsers.py` | Every extracted fact the prompts are built from | free |
| `tests/test_validation.py` | Grounding, confidence, OpenAPI, real test execution | free |
| `tests/test_api.py` | Routing, governance, repair loop, response shape | free |
| `tests/test_adversarial.py` | Injection, secrets, fabrication, huge input, contradictions | free |
| `tests/test_mapreduce.py` | Every chunk is sent; evidence scoped to what each call saw | free |
| `tests/test_e2e_local.py` | Real HTTP against a local OpenAI-compatible mock | free |
| `tests/markdown.test.mjs` | The renderer, including XSS | free |
| `evals/` | **Model output quality against a golden set** | real API calls |

The split matters. The `tests/` suites prove the system behaves correctly; they
say nothing about whether the model's analysis is any good. Only `evals/` does
that, which is why it exists and why it is run separately.
