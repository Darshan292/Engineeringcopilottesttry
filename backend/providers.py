"""What differs between OpenAI-compatible providers, in one place.

Every provider in this space speaks the same chat-completions wire format, which
is why the client is a thin POST. What they do *not* agree on is everything
around the edges of a request, and those edges are where the failures live:

- **How they say "come back later".** Some send `retry-after` as a duration.
  OpenRouter's free-tier 429 often sends neither that nor a delay in the body,
  and communicates the moment the window reopens as `X-RateLimit-Reset` -- a
  Unix timestamp in milliseconds. Reading one as the other is not a small bug:
  a millisecond timestamp parsed as a duration is a sleep of roughly fifty
  thousand years, so `seconds_until_reset` sanity-checks the magnitude rather
  than trusting the profile's flag.
- **What they actually ration.** Some ration tokens per minute. OpenRouter's
  free tier rations *requests* per minute and per day and reports no token
  ceiling at all. Budgeting a request-limited provider against a token
  allowance produces refusals that have nothing to do with the real limit.
- **What their model list looks like.** The same field is `context_window` in
  one and `context_length` in another, sometimes nested under `top_provider`.

Hardcoding any of this to one vendor is how this application once told a user
their 131,072-token model had no room for input. A profile makes the differences
data, so a new endpoint is a table entry rather than a hunt through the client.

OpenRouter is the provider this application is built around and the default.
`generic` exists for an OpenAI-compatible server you run yourself -- Ollama,
llama.cpp, vLLM, LM Studio -- and assumes nothing about rationing. Selection is
by `LLM_PROVIDER`, else inferred from the base URL's host, else OpenRouter.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse


@dataclass(frozen=True)
class Provider:
    """How to talk to one OpenAI-compatible endpoint, and what to expect back."""

    name: str
    label: str
    base_url: str
    # Environment variables holding the key, most specific first.
    key_envs: tuple[str, ...]
    console_keys_url: str

    # --- rate limiting -----------------------------------------------------
    # Header naming when the current window reopens, and whether its value is a
    # point in time or a number of seconds from now.
    reset_header: str | None = None
    reset_is_absolute: bool = False
    # Header naming the per-minute token ceiling, when the provider has one.
    token_limit_header: str | None = "x-ratelimit-limit-tokens"

    # Free-tier defaults, used only when the operator has not set their own.
    # `tokens_per_minute = 0` means this provider does not ration tokens, so the
    # budget should fall back to the model's context window instead.
    default_requests_per_minute: int = 20
    default_requests_per_day: int = 500
    default_tokens_per_minute: int = 0
    default_tokens_per_day: int = 0

    # What to wait when a 429 arrives stating no delay at all. Anything is
    # better than treating silence as "give up", which is how a rate limit
    # turned back into a failed request.
    blind_retry_seconds: float = 20.0

    # Sent on every request. OpenRouter uses these for attribution and they are
    # harmless elsewhere.
    extra_headers: dict[str, str] = field(default_factory=dict)

    # Hint text when a model turns out not to exist on this provider.
    model_hint: str = ""

    def public(self) -> dict:
        return {
            "provider": self.name,
            "label": self.label,
            "base_url": self.base_url,
            "rations_tokens": bool(self.default_tokens_per_minute),
            "default_requests_per_minute": self.default_requests_per_minute,
            "default_requests_per_day": self.default_requests_per_day,
        }


OPENROUTER = Provider(
    name="openrouter",
    label="OpenRouter",
    base_url="https://openrouter.ai/api/v1",
    key_envs=("OPENROUTER_API_KEY", "LLM_API_KEY"),
    console_keys_url="https://openrouter.ai/settings/keys",
    # A Unix timestamp in milliseconds. The parser sanity-checks the magnitude
    # as well, because getting this wrong is catastrophic rather than merely
    # wrong, and a flag in a table is exactly the kind of thing that goes stale.
    reset_header="x-ratelimit-reset",
    reset_is_absolute=True,
    # OpenRouter rations requests, not tokens, and sends no token ceiling.
    token_limit_header=None,
    # Free models: 20 requests/minute, and 50/day below $10 of lifetime credit
    # (1,000/day above it). The daily figure is the one that bites, so it is the
    # conservative default -- raise RATE_LIMIT_PER_DAY once you have credit.
    default_requests_per_minute=20,
    default_requests_per_day=50,
    default_tokens_per_minute=0,
    default_tokens_per_day=0,
    # The free-tier per-minute 429 carries no Retry-After and sometimes no
    # reset header either. The window is a minute, so wait out a sensible slice
    # of one rather than abandoning the request.
    blind_retry_seconds=20.0,
    extra_headers={
        # Optional, used by OpenRouter for attribution. Overridable so a real
        # deployment can identify itself.
        "HTTP-Referer": os.getenv("LLM_SITE_URL", "http://localhost:8000"),
        "X-Title": os.getenv("LLM_SITE_NAME", "Engineering Copilot"),
    },
    model_hint=(
        "Model IDs are 'vendor/model' and free variants end in ':free', e.g. "
        "'deepseek/deepseek-chat:free'. Open GET /api/models for the live list."
    ),
)

# Anything else OpenAI-compatible: a local Ollama, llama.cpp, vLLM, LM Studio,
# or a provider this table has never heard of. Assume nothing about rationing;
# a local model has no quota and a cloud one will say so in its headers.
GENERIC = Provider(
    name="generic",
    label="OpenAI-compatible endpoint",
    base_url="http://127.0.0.1:11434/v1",
    key_envs=("LLM_API_KEY", "OPENROUTER_API_KEY"),
    console_keys_url="",
    reset_header=None,
    reset_is_absolute=False,
    token_limit_header="x-ratelimit-limit-tokens",
    default_requests_per_minute=60,
    default_requests_per_day=100_000,
    default_tokens_per_minute=0,
    default_tokens_per_day=0,
    blind_retry_seconds=15.0,
    model_hint="Check that LLM_BASE_URL points at an OpenAI-compatible server.",
)

PROVIDERS: dict[str, Provider] = {p.name: p for p in (OPENROUTER, GENERIC)}

# Host fragment -> provider, for inferring from a base URL alone.
_HOST_HINTS: tuple[tuple[str, Provider], ...] = (
    ("openrouter.ai", OPENROUTER),
)


def _configured_base_url() -> str | None:
    """The base URL the operator set, under any of the accepted names."""
    for name in ("LLM_BASE_URL", "OPENAI_BASE_URL"):
        raw = os.getenv(name)
        if raw and raw.strip():
            return raw.strip().rstrip("/")
    return None


def detect_provider(base_url: str | None = None) -> Provider:
    """Which provider profile applies.

    Explicit `LLM_PROVIDER` wins, then the base URL's host, then OpenRouter,
    which is what this application is built around.
    """
    named = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if named in PROVIDERS:
        return PROVIDERS[named]

    url = base_url or _configured_base_url()
    if url:
        host = (urlparse(url).hostname or "").lower()
        for fragment, provider in _HOST_HINTS:
            if host == fragment or host.endswith("." + fragment):
                return provider
        # A URL pointing somewhere unrecognised gets the conservative profile
        # rather than OpenRouter's, whose rate-limit shapes would be wrong.
        if host:
            return GENERIC
    return OPENROUTER


def resolve_base_url(provider: Provider) -> str:
    """The endpoint to call: what was configured, else the profile's own."""
    return _configured_base_url() or provider.base_url


def resolve_api_key(provider: Provider) -> str:
    """The first key present among the names this provider accepts."""
    for name in provider.key_envs:
        raw = os.getenv(name)
        if raw and raw.strip():
            return raw.strip()
    return ""


def api_key_env_name(provider: Provider) -> str:
    """The variable to name when telling someone the key is missing."""
    return provider.key_envs[0]


# Unix timestamps in milliseconds are above this; anything smaller is a
# duration. The check is deliberately independent of the profile's flag,
# because a stale flag here costs a sleep measured in millennia.
_MS_TIMESTAMP_FLOOR = 1e11


def seconds_until_reset(raw: str | None, *, absolute: bool, now: float | None = None) -> float | None:
    """Turn a reset header into "seconds from now", whatever shape it came in.

    Handles a duration (`"4.3"`, `"250ms"`, `"1m"`) and an absolute instant in
    Unix seconds or milliseconds, and refuses to believe a value that is neither
    rather than sleeping on it.
    """
    if not raw:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None

    now = time.time() if now is None else now

    # Duration with an explicit unit is unambiguous, so honour it first.
    for suffix, scale in (("ms", 0.001), ("s", 1.0), ("m", 60.0), ("h", 3600.0)):
        if text.endswith(suffix):
            head = text[: -len(suffix)].strip()
            try:
                return max(0.0, float(head) * scale)
            except ValueError:
                return None

    try:
        value = float(text)
    except ValueError:
        return None

    # A bare number: absolute or relative depends on the provider, but the
    # magnitude is the stronger signal and overrules a stale profile either way.
    looks_absolute = value >= _MS_TIMESTAMP_FLOOR or (absolute and value > 1e8)
    if looks_absolute:
        # Milliseconds if it is far too large to be seconds.
        seconds = value / 1000.0 if value >= _MS_TIMESTAMP_FLOOR else value
        return max(0.0, seconds - now)
    if absolute and value > now:
        return max(0.0, value - now)
    return max(0.0, value)
