"""Pipeline primitives: structured calls, repair loops, and tracing.

Two things live here.

**Structured calling.** The model is asked for JSON matching a schema and the
response is validated with pydantic. When validation fails the error is fed
back verbatim -- field path and all -- and the call is retried. Two repair
attempts, then give up honestly rather than shipping something unvalidated.
This is bounded on purpose: an unbounded repair loop against a rate-limited
free tier is a way to burn a daily quota on one bad request.

Repair is not only for schema errors. Domain validators feed the same loop, so
"your generated tests fail with this assertion error" and "this OpenAPI
document is missing the routes the parser found" are handled by the same
mechanism as a missing field.

**Tracing.** Every stage records its name, duration and a small result summary.
The trace is returned with the response, so a slow or wrong result can be
attributed to a stage rather than guessed at. With a request ID attached it is
what makes concurrent requests debuggable at all.
"""

from __future__ import annotations

import inspect
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..core.tokens import estimate_messages_tokens
from ..groq_client import GroqError, complete

T = TypeVar("T", bound=BaseModel)

MAX_REPAIR_ATTEMPTS = 2


# --- tracing --------------------------------------------------------------


@dataclass
class Stage:
    name: str
    duration_ms: int
    summary: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class Trace:
    request_id: str
    stages: list[Stage] = field(default_factory=list)
    upstream_calls: int = 0
    total_tokens: int = 0
    # Time spent waiting for token allowance rather than working. Reported so
    # a slow request is explicable rather than mysterious.
    waited_seconds: float = 0.0

    def record(self, name: str, started: float, summary: dict | None = None, error: str | None = None) -> None:
        self.stages.append(
            Stage(
                name=name,
                duration_ms=int((time.perf_counter() - started) * 1000),
                summary=summary or {},
                error=error,
            )
        )

    def public(self) -> dict:
        return {
            "request_id": self.request_id,
            "upstream_calls": self.upstream_calls,
            "total_tokens": self.total_tokens,
            "waited_seconds": round(self.waited_seconds, 1),
            "stages": [
                {"name": s.name, "duration_ms": s.duration_ms, **({"error": s.error} if s.error else {}), **s.summary}
                for s in self.stages
            ],
        }


class timed:
    """Context manager that records a stage on exit, even when it raises."""

    def __init__(self, trace: Trace, name: str):
        self.trace = trace
        self.name = name
        self.summary: dict = {}
        self._started = 0.0

    def __enter__(self) -> "timed":
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.trace.record(
            self.name,
            self._started,
            self.summary,
            error=f"{exc_type.__name__}: {exc}" if exc_type else None,
        )
        return False


# --- JSON extraction ------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.S)


def extract_json(text: str) -> tuple[dict | None, str | None]:
    """Recover a JSON object from a model response. Returns (obj, error)."""
    if not text or not text.strip():
        return None, "The model returned an empty response."

    candidates: list[str] = []

    # Fenced block first: the most common wrapper despite instructions.
    for match in _FENCE_RE.finditer(text):
        candidates.append(match.group(1))

    stripped = text.strip()
    candidates.append(stripped)

    # Outermost brace pair, for a response with prose on either side.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    last_error = "No JSON object found in the response."
    for candidate in candidates:
        body = candidate.strip()
        if not body:
            continue
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            last_error = f"JSON parse error: {exc.msg} at line {exc.lineno} column {exc.colno}"
            # Trailing commas are the single most common malformation; one
            # targeted retry is cheaper than a whole repair round-trip.
            repaired = re.sub(r",(\s*[}\]])", r"\1", body)
            if repaired != body:
                try:
                    parsed = json.loads(repaired)
                except json.JSONDecodeError:
                    continue
            else:
                continue

        if isinstance(parsed, dict):
            return parsed, None
        last_error = f"Top-level JSON is {type(parsed).__name__}, expected an object."

    return None, last_error


def format_validation_error(exc: ValidationError) -> str:
    """Turn pydantic's structure into instructions the model can act on."""
    lines: list[str] = []
    for error in exc.errors()[:12]:
        location = ".".join(str(p) for p in error["loc"]) or "(root)"
        lines.append(f"  - field `{location}`: {error['msg']} (got {error.get('input')!r:.80})")
    return "The JSON did not match the required schema:\n" + "\n".join(lines)


# --- structured call with repair -----------------------------------------


@dataclass
class StructuredResult:
    value: Any
    attempts: int
    repairs: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    model: str = ""
    elapsed_ms: int = 0
    raw_last: str = ""

    def merge(self, other: "StructuredResult") -> "StructuredResult":
        """Fold another call's cost into this one, for multi-call runs."""
        merged = StructuredResult(
            value=other.value,
            attempts=self.attempts + other.attempts,
            repairs=self.repairs + other.repairs,
            usage={
                key: (self.usage.get(key, 0) or 0) + (other.usage.get(key, 0) or 0)
                for key in {"prompt_tokens", "completion_tokens", "total_tokens"}
            },
            model=other.model or self.model,
            elapsed_ms=self.elapsed_ms + other.elapsed_ms,
            raw_last=other.raw_last,
        )
        return merged


async def call_structured(
    system_prompt: str,
    user_content: str,
    schema: type[T],
    *,
    trace: Trace,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stage_name: str = "llm_structured",
    extra_validators: list | None = None,
    client_key: str = "local",
) -> StructuredResult:
    """Call the model, validate against `schema`, repair on failure.

    `extra_validators` are callables taking the validated object and returning
    an error string (or empty). They participate in the same repair loop, which
    is how test-execution failures and OpenAPI errors get a second chance.
    They may be async: a validator that shells out (running generated tests,
    for instance) must not do so on the event loop, so it returns a coroutine
    and is awaited here.
    """
    messages_user = user_content
    repairs: list[str] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    elapsed = 0
    last_raw = ""
    used_model = model or ""

    from ..governance import await_token_budget, token_limiter

    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 2):
        # Check the token budget before each attempt, including repairs. A
        # repair costs as much as the original call, so a loop that ignores the
        # budget turns one oversized request into an upstream 429 naming a
        # limit the caller has never seen.
        projected = estimate_messages_tokens(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": messages_user}]
        ) + (max_tokens or 0)
        # Wait for budget rather than failing. On a rate-limited account a
        # multi-part analysis genuinely takes minutes, and a complete answer
        # that arrives slowly is worth more than a fast refusal.
        waited, reservation = await await_token_budget(
            client_key,
            projected,
            stage=f"{stage_name} attempt {attempt}" if attempt > 1 else stage_name,
            waited_so_far=trace.waited_seconds,
        )
        trace.waited_seconds += waited
        if waited:
            trace.record(
                f"{stage_name}.wait{attempt}",
                time.perf_counter() - waited,
                {"waited_seconds": round(waited, 1), "reason": "token allowance"},
            )

        with timed(trace, f"{stage_name}.attempt{attempt}") as stage:
            result = await complete(
                system_prompt,
                messages_user,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
            )
            trace.upstream_calls += 1
            # The pacer reserved `projected`; replace it with what the call
            # actually cost. Settling both ways matters: charging the excess
            # when the estimate was low keeps the limiter honest, and releasing
            # the surplus when it was high stops a conservative output reserve
            # from permanently eating a slice of every minute.
            actual = int(result["usage"].get("total_tokens") or projected)
            token_limiter.settle(reservation, actual)
            last_raw = result["text"]
            used_model = result["model"]
            elapsed += result["elapsed_ms"]
            for key in total_usage:
                total_usage[key] += result["usage"].get(key) or 0
            trace.total_tokens = total_usage["total_tokens"]
            stage.summary = {
                "finish_reason": result["finish_reason"],
                "tokens": result["usage"].get("total_tokens"),
            }

            payload, parse_error = extract_json(result["text"])
            if parse_error:
                stage.summary["outcome"] = "unparseable"
                feedback = (
                    f"{parse_error}\n\nReturn ONLY a single JSON object. No prose before or "
                    f"after it, no Markdown code fence."
                )
            else:
                try:
                    value = schema.model_validate(payload)
                except ValidationError as exc:
                    stage.summary["outcome"] = "schema_invalid"
                    feedback = format_validation_error(exc)
                else:
                    domain_errors = []
                    for validator in extra_validators or []:
                        message = validator(value)
                        if inspect.isawaitable(message):
                            message = await message
                        if message:
                            domain_errors.append(message)
                    if not domain_errors:
                        stage.summary["outcome"] = "valid"
                        return StructuredResult(
                            value=value,
                            attempts=attempt,
                            repairs=repairs,
                            usage=total_usage,
                            model=used_model,
                            elapsed_ms=elapsed,
                            raw_last=last_raw,
                        )
                    stage.summary["outcome"] = "domain_invalid"
                    feedback = "\n\n".join(domain_errors)

        if attempt > MAX_REPAIR_ATTEMPTS:
            raise GroqError(
                f"The model did not produce a valid response after {attempt} attempts. "
                f"Last problem: {feedback.splitlines()[0][:200]}",
                status=502,
                hint=(
                    "This usually means the configured model struggles with structured output. "
                    "Try GROQ_MODEL=openai/gpt-oss-120b, which follows JSON schemas more reliably "
                    "than smaller models."
                ),
            )

        # Keep the first line that actually says something: the header line of
        # a schema error names no field, and the field path is the whole point.
        informative = [
            line.strip() for line in feedback.splitlines()
            if line.strip() and not line.rstrip().endswith(":")
        ]
        repairs.append((informative[0] if informative else feedback.splitlines()[0])[:240])
        messages_user = (
            f"{user_content}\n\n"
            f"--- YOUR PREVIOUS ANSWER WAS REJECTED ---\n"
            f"{feedback}\n\n"
            f"Produce a corrected JSON object. Fix only what is listed above; keep everything "
            f"else you had right. Return the complete object, not a patch."
        )

    raise AssertionError("unreachable")  # pragma: no cover
