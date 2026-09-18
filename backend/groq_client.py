"""Thin async client for Groq's OpenAI-compatible chat completions endpoint.

Raw HTTP via httpx rather than the `groq` SDK: one fewer dependency, and the
wire format is stable and OpenAI-compatible, so this keeps working across SDK
churn. The value this module adds over a bare POST is error translation --
turning Groq's HTTP failures into messages that tell you what to actually do.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from .config import DEFAULT_MODEL, RETIRED_MODELS, settings


class GroqError(Exception):
    """A Groq call failed in a way the user needs to read.

    `status` is the HTTP status we should surface to the browser; `hint` is the
    actionable remediation line shown under the error in the UI.
    """

    def __init__(self, message: str, *, status: int = 502, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.hint = hint


def _auth_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }


def _require_key() -> None:
    if not settings.has_api_key:
        raise GroqError(
            "GROQ_API_KEY is not set, so there is nothing to authenticate with.",
            status=503,
            hint=(
                "Create a free key at https://console.groq.com/keys, then "
                "`cp .env.example .env`, paste the key into it, and restart the server."
            ),
        )


def _extract_api_error(response: httpx.Response) -> str:
    """Pull Groq's own error string out of the body, falling back to raw text."""
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


def _translate_http_error(response: httpx.Response, model: str) -> GroqError:
    """Map a Groq HTTP failure onto a message that says what to fix."""
    detail = _extract_api_error(response)
    code = response.status_code

    if code in (401, 403):
        return GroqError(
            f"Groq rejected the API key ({code}): {detail}",
            status=502,
            hint="Check GROQ_API_KEY in .env. Regenerate it at https://console.groq.com/keys if unsure.",
        )

    if code == 404 or "model_not_found" in detail or "does not exist" in detail.lower():
        retired = RETIRED_MODELS.get(model)
        hint = (
            f"'{model}' is retired: {retired} Set GROQ_MODEL={DEFAULT_MODEL} in .env and restart."
            if retired
            else (
                f"Groq does not serve '{model}' on this account. "
                f"Open GET /api/models to see what your key can actually call, "
                f"then set GROQ_MODEL in .env."
            )
        )
        return GroqError(f"Model '{model}' is unavailable: {detail}", status=502, hint=hint)

    if code == 429:
        retry_after = response.headers.get("retry-after")
        wait = f" Retry after {retry_after}s." if retry_after else ""
        return GroqError(
            f"Groq rate limit hit: {detail}",
            status=429,
            hint=(
                f"The free tier is roughly 30 requests/min and 1,000/day per model.{wait} "
                f"Wait a moment, or switch GROQ_MODEL to a different free-tier model to "
                f"use a separate bucket."
            ),
        )

    if code == 413 or "too large" in detail.lower() or "context" in detail.lower():
        return GroqError(
            f"Input too large for the model's context window: {detail}",
            status=413,
            hint="Paste a smaller excerpt, or lower GROQ_MAX_TOKENS to leave more room for input.",
        )

    if code >= 500:
        return GroqError(
            f"Groq returned a server error ({code}): {detail}",
            status=502,
            hint="This is upstream, not your config. Retry in a few seconds.",
        )

    return GroqError(f"Groq request failed ({code}): {detail}", status=502)


async def complete(
    system_prompt: str,
    user_content: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Run one chat completion. Returns the text plus usage/latency metadata."""
    _require_key()

    chosen_model = (model or settings.groq_model).strip()
    body = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": settings.temperature if temperature is None else temperature,
        "max_tokens": settings.max_tokens if max_tokens is None else max_tokens,
        "stream": False,
    }

    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=settings.timeout_seconds) as client:
            response = await client.post(
                f"{settings.groq_base_url}/chat/completions",
                headers=_auth_headers(),
                json=body,
            )
    except httpx.TimeoutException as exc:
        raise GroqError(
            f"Groq did not respond within {settings.timeout_seconds:.0f}s.",
            status=504,
            hint="Try a shorter input, or raise GROQ_TIMEOUT_SECONDS in .env.",
        ) from exc
    except httpx.HTTPError as exc:
        raise GroqError(
            f"Could not reach Groq: {exc}",
            status=502,
            hint="Check your network connection and that GROQ_BASE_URL is correct.",
        ) from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if response.status_code >= 400:
        raise _translate_http_error(response, chosen_model)

    try:
        payload = response.json()
        choice = payload["choices"][0]
        text = choice["message"]["content"] or ""
    except (ValueError, KeyError, IndexError) as exc:
        raise GroqError(
            f"Groq returned a response this app could not parse: {exc}",
            status=502,
            hint="This usually means the endpoint is not OpenAI-compatible. Check GROQ_BASE_URL.",
        ) from exc

    if not text.strip():
        raise GroqError(
            "Groq returned an empty completion.",
            status=502,
            hint=(
                "Often means max_tokens was consumed by reasoning, or the input was "
                "filtered. Try a smaller input or raise GROQ_MAX_TOKENS."
            ),
        )

    usage = payload.get("usage") or {}
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
    }


async def list_models() -> list[dict[str, Any]]:
    """Ask Groq what this key can actually call, right now.

    Model IDs churn -- this is the only trustworthy source, so the UI reads it
    rather than trusting any list baked into the code.
    """
    _require_key()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{settings.groq_base_url}/models", headers=_auth_headers()
            )
    except httpx.HTTPError as exc:
        raise GroqError(f"Could not reach Groq: {exc}", status=502) from exc

    if response.status_code >= 400:
        raise _translate_http_error(response, settings.groq_model)

    try:
        data = response.json().get("data", [])
    except ValueError as exc:
        raise GroqError("Groq's model list was not valid JSON.", status=502) from exc

    models = [
        {
            "id": m.get("id"),
            "owned_by": m.get("owned_by"),
            "context_window": m.get("context_window"),
            "active": m.get("active", True),
        }
        for m in data
        if m.get("id")
    ]
    models.sort(key=lambda m: str(m["id"]))
    return models
