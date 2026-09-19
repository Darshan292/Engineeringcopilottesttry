"""Thin async client for an OpenAI-compatible chat completions endpoint.

Raw HTTP via httpx rather than a vendor SDK: one fewer dependency, and the wire
format is stable and OpenAI-compatible, so this keeps working across SDK churn.
The value this module adds over a bare POST is threefold -- translating HTTP
failures into messages that say what to fix, waiting out a rate limit instead of
surfacing it, and refusing to send anything the billing guard has not cleared as
free.

What differs between endpoints lives in `providers.py`, not here.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

import httpx

from .config import RETIRED_MODELS, settings
from .billing import guard, price_check
from .providers import api_key_env_name, detect_provider, seconds_until_reset

log = logging.getLogger("copilot.llm")

# A 429 is the provider telling us when to come back, not that the work cannot
# be done. Waiting it out is the whole difference between "this application
# needs a paid tier" and "this application is slower on a free tier", and the
# second is the product this is meant to be.
MAX_RATE_LIMIT_RETRIES = 6

# Cap on a single wait between retries. A provider asking for longer than this
# is describing a quota problem rather than a burst, and the caller is better
# told than left on a hanging request.
MAX_SINGLE_RETRY_WAIT = 65.0


async def _sleep(seconds: float) -> None:
    """Indirection so tests can observe the wait without serving it.

    Patching `asyncio.sleep` itself would reach every other coroutine in the
    process; this keeps the seam local to the client.
    """
    await asyncio.sleep(seconds)


class LLMError(Exception):
    """An upstream call failed in a way the user needs to read.

    `status` is the HTTP status we should surface to the browser; `hint` is the
    actionable remediation line shown under the error in the UI.
    """

    def __init__(self, message: str, *, status: int = 502, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.hint = hint


def _is_loopback(url: str) -> bool:
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"} or host.endswith(".localhost")


def _client() -> httpx.AsyncClient:
    """Build the HTTP client for the configured endpoint.

    A loopback base URL means a local OpenAI-compatible server (Ollama,
    llama.cpp, vLLM, a test double). Those must not be routed through an
    ambient HTTP_PROXY: on a machine with a corporate proxy configured, httpx
    would send the request to the proxy, which refuses to tunnel to localhost,
    and the app fails with a confusing 403 for a server running on the same
    machine. `trust_env=False` bypasses proxy resolution for that case only;
    remote traffic still honours the environment's proxy settings.
    """
    return httpx.AsyncClient(
        timeout=settings.timeout_seconds,
        trust_env=not _is_loopback(settings.base_url),
    )


def _auth_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
        # Whatever this provider wants alongside the key. OpenRouter uses
        # HTTP-Referer and X-Title for attribution; others ignore them.
        **_provider().extra_headers,
    }


def _provider():
    """The active provider profile, derived from the configured base URL.

    Resolved per call rather than captured at import, so a test or a reload that
    repoints the endpoint is picked up instead of silently using the old rules.
    """
    return detect_provider(settings.base_url)


def _require_key() -> None:
    if not settings.has_api_key:
        provider = _provider()
        key_name = api_key_env_name(provider)
        where = (
            f"Create a free key at {provider.console_keys_url}, then "
            if provider.console_keys_url
            else ""
        )
        raise LLMError(
            f"{key_name} is not set, so there is nothing to authenticate with.",
            status=503,
            hint=(
                f"{where}`cp .env.example .env`, paste the key into it, and restart "
                f"the server."
            ),
        )


def _extract_api_error(response: httpx.Response) -> str:
    """Pull the provider's own error string out of the body, else raw text."""
    try:
        payload = response.json()
    except ValueError:
        return (response.text or "").strip()[:500] or f"HTTP {response.status_code}"
    err = payload.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err.get("code") or err)[:500]
    if isinstance(err, str):
        return err[:500]
    return str(payload)[:500]


def _translate_http_error(response: httpx.Response, model: str) -> LLMError:
    """Map an HTTP failure onto a message that says what to fix."""
    detail = _extract_api_error(response)
    code = response.status_code

    if code in (401, 403):
        return LLMError(
            f"{_provider().label} rejected the API key ({code}): {detail}",
            status=502,
            hint=(
                f"Check {api_key_env_name(_provider())} in .env. "
                f"Regenerate it at {_provider().console_keys_url} if unsure."
            ),
        )

    if code == 404 or "model_not_found" in detail or "does not exist" in detail.lower():
        retired = RETIRED_MODELS.get(model)
        hint = (
            f"'{model}' is retired: {retired} Pick another from GET /api/models "
            f"and set LLM_MODEL in .env."
            if retired
            else (
                f"{_provider().label} does not serve '{model}' on this account. "
                f"Open GET /api/models to see what your key can actually call, "
                f"then set LLM_MODEL in .env. {_provider().model_hint}"
            )
        )
        return LLMError(f"Model '{model}' is unavailable: {detail}", status=502, hint=hint)

    if code == 429:
        # Only reached once the retries below are exhausted, so the advice is
        # about quota rather than about waiting.
        routed = _routed_model(detail)
        routing_note = (
            f" You selected '{model}', which routed to '{routed}'; the limit that bound is "
            f"that model's, not the one you configured."
            if routed and routed != model
            else ""
        )
        return LLMError(
            f"{_provider().label} rate limit hit and did not clear after waiting: {detail}",
            status=429,
            hint=(
                f"This request waited for the allowance to roll and the limit was still "
                f"reached.{routing_note} Lower TOKEN_LIMIT_PER_MINUTE so this app paces itself "
                f"below the real ceiling, wait a minute, or switch LLM_MODEL to a model with a "
                f"separate bucket."
            ),
        )

    if code == 413 or "too large" in detail.lower() or "context" in detail.lower():
        return LLMError(
            f"Input too large for the model's context window: {detail}",
            status=413,
            hint="Paste a smaller excerpt, or lower LLM_MAX_TOKENS to leave more room for input.",
        )

    if code >= 500:
        return LLMError(
            f"{_provider().label} returned a server error ({code}): {detail}",
            status=502,
            hint="This is upstream, not your config. Retry in a few seconds.",
        )

    return LLMError(f"{_provider().label} request failed ({code}): {detail}", status=502)



def _learn_from_rate_limit(model: str, detail: str, body: dict, max_tokens: int | None) -> None:
    """Take the provider's own arithmetic from a 429 and believe it.

    The body names the limit that bound, the model that bound it, and what this
    call was counted as costing. That last figure is the valuable one: for an
    agentic or routing model the wire cost is a multiple of the text we sent,
    because the provider prepends tool schemas and internal instructions we
    never see. Our estimate cannot converge on that from successful calls,
    because while the estimate is too low the calls are not successful.
    """
    try:
        from .config import is_routing_model
        from .core.tokens import calibrator, estimate_messages_tokens
        from .governance import adopt_limit_from_rate_limit_error

        learned = adopt_limit_from_rate_limit_error(model, detail)
        if not learned:
            return

        requested = learned.get("requested")
        routed_to = learned.get("routed_to")
        # Only attribute the cost when it is genuinely this model's. A 429 that
        # names a different model is the truth about `model` when `model` is a
        # router that dispatched to it, and is something else entirely -- a
        # shared organisation bucket, a mislabelled proxy -- when it is not.
        # Believing it in that second case inflates the estimate for a model
        # that never paid the cost, and the inflated system reserve can wipe out
        # the whole input budget for every later request in the process.
        attributable = (
            not routed_to
            or routed_to == model
            or is_routing_model(model)
        )
        if requested and attributable:
            ours = estimate_messages_tokens(body.get("messages") or []) + (max_tokens or 0)
            if ours > 0:
                calibrator.observe(model, ours, requested)
        elif requested:
            log.info(
                "not calibrating %s from a limit reached by %s; %s is not a routing model",
                model, routed_to, model,
            )
        if learned.get("routed_to") and learned["routed_to"] != model:
            log.info("%s routed to %s; limit %s applies to the latter",
                     model, learned["routed_to"], learned.get("limit"))
    except Exception:  # never let learning break the call path
        pass



def _routed_model(detail: str) -> str | None:
    """The model a 429 says actually ran, which may not be the one requested."""
    match = re.search(r"for model `([^`]+)`", detail or "")
    return match.group(1) if match else None


def _retry_delay(response: httpx.Response, detail: str, attempt: int) -> float:
    """How long to wait before trying again. Always answers.

    Four sources, most authoritative first: the `retry-after` header, the
    provider's reset header, the sentence in the error body ("Please try again
    in 4.303999999s"), and finally a backoff derived from the provider profile.

    That last fallback is the point. Returning "no idea" used to mean giving up,
    which is fine for a provider that always states a delay and wrong for one
    that does not: OpenRouter's free-tier per-minute 429 carries no Retry-After
    at all, so silence would have reproduced exactly the bug this retry loop was
    written to fix -- on a different provider.

    The reset header is parsed through the provider profile because its units
    are not agreed: some state a duration, OpenRouter an absolute Unix
    millisecond timestamp. `seconds_until_reset` also sanity-checks the
    magnitude, so mislabelling one as the other cannot produce an absurd sleep.
    """
    provider = _provider()

    header = response.headers.get("retry-after")
    if header:
        try:
            return max(0.0, float(header.strip()))
        except ValueError:
            pass

    if provider.reset_header:
        from_reset = seconds_until_reset(
            response.headers.get(provider.reset_header),
            absolute=provider.reset_is_absolute,
        )
        if from_reset is not None:
            return from_reset

    match = re.search(r"try again in ([\d.]+)\s*(ms|s)?", detail or "", re.IGNORECASE)
    if match:
        value = float(match.group(1))
        return value / 1000 if match.group(2) == "ms" else value

    # Nothing stated. Back off geometrically from the profile's blind wait so a
    # provider that never says anything still gets a few honest attempts.
    return provider.blind_retry_seconds * (1.5 ** (attempt - 1))


async def complete(
    system_prompt: str,
    user_content: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
    max_wait_seconds: float | None = None,
) -> dict[str, Any]:
    """Run one chat completion. Returns the text plus usage/latency metadata.

    `json_mode` sets the OpenAI-compatible `response_format` so the server
    constrains decoding to valid JSON. It removes the whole class of "the model
    wrote a sentence before the object" failures at the source rather than
    leaving them to the repair loop. Servers that do not support it reject the
    request with a 400, which is retried once without the flag.
    """
    _require_key()

    chosen_model = (model or settings.model).strip()
    if not chosen_model:
        provider = _provider()
        raise LLMError(
            f"No model is configured, so there is nothing to send the request to.",
            status=503,
            hint=(
                f"Set LLM_MODEL in your .env. {provider.label} ships no default here because "
                f"its catalogue changes faster than any list compiled into this application. "
                f"GET /api/models shows what your key can call right now; free models are "
                f"flagged. {provider.model_hint}"
            ),
        )
    body: dict[str, Any] = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": settings.temperature if temperature is None else temperature,
        "max_tokens": settings.max_tokens if max_tokens is None else max_tokens,
        "stream": False,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    # Layer 1: nothing is callable until the provider's own price list has shown
    # it to be free. Raised before the socket is opened, so a misconfiguration
    # costs nothing at all.
    guard.assert_free(chosen_model)
    # Layer 2: ask the provider to enforce the same ceiling on its side.
    body.update(guard.request_guard_fields(_provider().name))

    started = time.perf_counter()
    rate_limit_waits = 0.0
    rate_limit_retries = 0

    async def post() -> httpx.Response:
        try:
            async with _client() as client:
                return await client.post(
                    f"{settings.base_url}/chat/completions",
                    headers=_auth_headers(),
                    json=body,
                )
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"{_provider().label} did not respond within {settings.timeout_seconds:.0f}s.",
                status=504,
                hint="Try a shorter input, or raise LLM_TIMEOUT_SECONDS in .env.",
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Could not reach {_provider().label}: {exc}",
                status=502,
                hint="Check your network connection and that LLM_BASE_URL is correct.",
            ) from exc

    # A rate limit is a "come back shortly", not a refusal. The provider states
    # when, so the loop waits that long and tries again instead of turning a
    # four-second pause into a failed request. Each 429 also teaches us the real
    # ceiling and the real cost of this call, which is the only way an estimate
    # for a routing or agentic model can ever improve -- the request never
    # succeeds, so no usage is ever reported to learn from.
    if max_wait_seconds is None:
        from .governance import MAX_TOTAL_WAIT_SECONDS

        budget_left = float(MAX_TOTAL_WAIT_SECONDS)
    else:
        budget_left = max(0.0, max_wait_seconds)
    while True:
        response = await post()

        try:
            from .governance import adopt_provider_limits

            adopt_provider_limits(response.headers)
        except Exception:
            pass

        if response.status_code != 429:
            break

        detail = _extract_api_error(response)
        _learn_from_rate_limit(chosen_model, detail, body, max_tokens)

        if rate_limit_retries >= MAX_RATE_LIMIT_RETRIES:
            break
        delay = _retry_delay(response, detail, rate_limit_retries + 1)
        # A provider asking for longer than one window is describing exhausted
        # quota, not a burst; waiting it out would hang the request for minutes
        # with nothing to show.
        delay = min(delay + 0.25, MAX_SINGLE_RETRY_WAIT)
        if delay > budget_left:
            break

        rate_limit_retries += 1
        rate_limit_waits += delay
        budget_left -= delay
        log.info(
            "rate limited by %s; waiting %.1fs and retrying (attempt %d of %d)",
            _routed_model(detail) or chosen_model, delay, rate_limit_retries, MAX_RATE_LIMIT_RETRIES,
        )
        await _sleep(delay)

    # Not every OpenAI-compatible server implements response_format. Drop it
    # and retry once rather than failing a request over an optional flag.
    if response.status_code == 400 and json_mode and "response_format" in (response.text or ""):
        body.pop("response_format", None)
        response = await post()

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if response.status_code >= 400:
        raise _translate_http_error(response, chosen_model)

    try:
        payload = response.json()
        choice = payload["choices"][0]
        text = choice["message"]["content"] or ""
    except (ValueError, KeyError, IndexError) as exc:
        raise LLMError(
            f"{_provider().label} returned a response this app could not parse: {exc}",
            status=502,
            hint="This usually means the endpoint is not OpenAI-compatible. Check LLM_BASE_URL.",
        ) from exc

    if not text.strip():
        raise LLMError(
            f"{_provider().label} returned an empty completion.",
            status=502,
            hint=(
                "Often means max_tokens was consumed by reasoning, or the input was "
                "filtered. Try a smaller input or raise LLM_MAX_TOKENS."
            ),
        )

    usage = payload.get("usage") or {}

    # Layer 3: believe the invoice over every assumption that preceded it. A
    # non-zero figure here means the price list and the provider-side ceiling
    # were both wrong, which is the case no amount of pre-flight checking can
    # rule out -- so it latches and the next call is refused.
    charged = guard.observe_usage(chosen_model, usage)

    # Close the calibration loop: compare what we estimated for this exact
    # payload against what the server actually charged, so future budgets for
    # this model converge on the truth.
    actual_prompt_tokens = usage.get("prompt_tokens")
    if actual_prompt_tokens:
        try:
            from .core.tokens import calibrator, estimate_messages_tokens

            calibrator.observe(
                chosen_model,
                estimate_messages_tokens(body["messages"]),
                int(actual_prompt_tokens),
            )
        except Exception:
            # Calibration is an optimisation; never fail a good response over it.
            pass

    return {
        "text": text.strip(),
        "model": payload.get("model", chosen_model),
        "finish_reason": choice.get("finish_reason"),
        "elapsed_ms": elapsed_ms,
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
        # Time spent held back by the provider rather than working, reported so
        # a slow request is explicable instead of mysterious.
        "rate_limited_seconds": round(rate_limit_waits, 2),
        "rate_limit_retries": rate_limit_retries,
        # Zero on a free model, and surfaced rather than dropped so "this cost
        # nothing" is something the caller can see rather than take on trust.
        "cost": charged,
    }


def _context_length_of(model: dict) -> int | None:
    """The model's context window, under whichever name this provider uses.

    Some say `context_window`, OpenRouter says `context_length` and also
    repeats it under `top_provider`. Accepting all of them costs three lines and
    removes a whole class of "why does every model show an unknown window".
    """
    for key in ("context_window", "context_length", "max_context_length"):
        value = model.get(key)
        if isinstance(value, int) and value > 0:
            return value
    nested = model.get("top_provider")
    if isinstance(nested, dict):
        value = nested.get("context_length")
        if isinstance(value, int) and value > 0:
            return value
    return None


def _vendor_of(model_id: str) -> str | None:
    """'deepseek' from 'deepseek/deepseek-chat:free'."""
    return model_id.split("/", 1)[0] if "/" in model_id else None


async def list_models() -> list[dict[str, Any]]:
    """Ask the provider what this key can actually call, right now.

    Model IDs churn -- this is the only trustworthy source, so the UI reads it
    rather than trusting any list baked into the code.
    """
    _require_key()
    try:
        async with _client() as client:
            response = await client.get(
                f"{settings.base_url}/models", headers=_auth_headers()
            )
    except httpx.HTTPError as exc:
        raise LLMError(f"Could not reach {_provider().label}: {exc}", status=502) from exc

    if response.status_code >= 400:
        raise _translate_http_error(response, settings.model)

    try:
        data = response.json().get("data", [])
    except ValueError as exc:
        raise LLMError(f"{_provider().label}'s model list was not valid JSON.", status=502) from exc

    from .core.tokens import MIN_USABLE_CONTEXT, non_chat_reason

    # The raw catalogue is the price list, so the billing guard is fed from it
    # here -- the one place it is fetched. Every model's verdict is recorded
    # before any of it is reshaped for the UI, so what the guard enforces and
    # what the list displays cannot drift apart.
    guard.load_catalogue(data)

    models = []
    for m in data:
        model_id = m.get("id")
        if not model_id:
            continue
        window = _context_length_of(m)
        reason = non_chat_reason(str(model_id))
        if reason is None and isinstance(window, int) and 0 < window < MIN_USABLE_CONTEXT:
            reason = f"context window of {window:,} tokens is too small for this application"
        models.append(
            {
                "id": model_id,
                "owned_by": m.get("owned_by") or _vendor_of(str(model_id)),
                "context_window": window,
                "active": m.get("active", True),
                # Marked rather than filtered: knowing a model is free is the
                # first thing anyone on a free tier needs, and the provider is
                # the only one who knows it.
                # The guard's verdict, not a second opinion. A label that can
                # disagree with what the app will actually allow is worse than
                # no label: it tells you a model is free right up until the
                # call is refused for costing money.
                #
                # This said exactly that and then called `price_check` anyway,
                # which is not the same function: it reads the price list and
                # knows nothing about routers that quote zero for themselves
                # while dispatching to paid models. `openrouter/auto` was
                # listed as free and refused when called.
                "free": guard.verdict(str(model_id))[0],
                # The provider lists every model the key can call, including
                # classifiers and speech models. Selecting one of those fails
                # deep in the pipeline, so mark them here where they are listed.
                "usable": reason is None,
                "unusable_reason": reason,
            }
        )
    models.sort(key=lambda m: str(m["id"]))

    # Treat what the provider reports as authoritative over the static table.
    # Windows change, models are retired and added, and being optimistically
    # wrong means a request that fails upstream after the whole local pipeline
    # has already run.
    try:
        from .core.tokens import registry

        registry.update(models)
    except Exception:
        pass

    return models


async def refresh_model_registry_if_stale() -> bool:
    """Opportunistically refresh runtime model metadata. Never raises.

    Called before budgeting. One cheap GET per hour keeps context windows
    accurate; a failure just leaves the previous values in place.
    """
    from .core.tokens import registry

    if not registry.should_refresh:
        return False
    if not settings.has_api_key:
        return False

    # Mark the attempt before making it, so a failure backs off rather than
    # repeating on every request for the rest of the process lifetime.
    registry.note_attempt()
    try:
        await list_models()
        return True
    except Exception:
        return False
