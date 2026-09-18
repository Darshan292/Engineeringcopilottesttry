"""Operational controls: model allowlist, concurrency, rate limiting, auth.

A local demo does not need these. A service someone points at a shared host
does, and the gap between the two is a single `--host 0.0.0.0`. Each control is
off or permissive by default so local use is unchanged, and each turns on with
one environment variable.

The rate limiter is deliberately the service's own, not a reliance on the
upstream provider's. Leaning on Groq's 429 means every over-limit request pays
a network round trip, consumes daily quota, and returns an error that mentions
a vendor the caller has no relationship with. Shedding locally is faster,
cheaper and gives a better error.
"""

from __future__ import annotations

import asyncio
import hmac
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def _env_list(name: str) -> list[str]:
    raw = os.getenv(name) or ""
    return [item.strip() for item in raw.split(",") if item.strip()]


class GovernanceError(Exception):
    def __init__(self, message: str, *, status: int = 403, hint: str | None = None, headers: dict | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.hint = hint
        self.headers = headers or {}


# --- model governance -----------------------------------------------------


@dataclass
class ModelPolicy:
    """Controls which models a caller may select.

    Default is permissive, because on a single-operator local tool the caller
    and the operator are the same person and blocking them helps nobody. Set
    `ALLOWED_MODELS` and per-request overrides are restricted to that list;
    set `ALLOW_MODEL_OVERRIDE=false` and they are refused outright.
    """

    allow_override: bool = field(default_factory=lambda: _env_bool("ALLOW_MODEL_OVERRIDE", True))
    allowed: list[str] = field(default_factory=lambda: _env_list("ALLOWED_MODELS"))
    allow_temperature_override: bool = field(
        default_factory=lambda: _env_bool("ALLOW_TEMPERATURE_OVERRIDE", True)
    )
    max_temperature: float = field(
        default_factory=lambda: float(os.getenv("MAX_TEMPERATURE") or 1.0)
    )

    def check_model(self, requested: str | None, default_model: str) -> str:
        if not requested or requested == default_model:
            return default_model
        if not self.allow_override:
            raise GovernanceError(
                "Per-request model selection is disabled on this deployment.",
                hint=f"The configured model is '{default_model}'. Unset ALLOW_MODEL_OVERRIDE to permit overrides.",
            )
        if self.allowed and requested not in self.allowed:
            raise GovernanceError(
                f"Model '{requested}' is not in this deployment's allowlist.",
                hint=f"Permitted models: {', '.join(self.allowed)}.",
            )
        return requested

    def check_temperature(self, requested: float | None) -> float | None:
        if requested is None:
            return None
        if not self.allow_temperature_override:
            raise GovernanceError(
                "Per-request temperature is disabled on this deployment.",
                hint="These are analysis tools; a low fixed temperature is intentional.",
            )
        if requested > self.max_temperature:
            raise GovernanceError(
                f"Temperature {requested} exceeds the maximum of {self.max_temperature}.",
                status=422,
                hint="High temperature degrades structured-output reliability.",
            )
        return requested

    def public(self) -> dict:
        return {
            "model_override_allowed": self.allow_override,
            "allowed_models": self.allowed or "any",
            "temperature_override_allowed": self.allow_temperature_override,
            "max_temperature": self.max_temperature,
        }


# --- rate limiting --------------------------------------------------------


class SlidingWindowLimiter:
    """Per-client sliding-window limiter, in memory, thread-safe.

    In-memory means per-process: it is a load-shedding control for a
    single-instance deployment, not a distributed quota. Said plainly here so
    nobody assumes it holds across replicas.
    """

    # Every distinct client key allocates two deques that were never reclaimed.
    # A service reachable by many addresses -- or one behind a proxy that
    # forwards a varying key -- grows that map without bound, which is a slow
    # memory leak in the component meant to protect against abuse.
    _MAX_TRACKED_CLIENTS = 10_000
    _SWEEP_EVERY = 500

    def __init__(self, per_minute: int, per_day: int):
        self.per_minute = per_minute
        self.per_day = per_day
        self._lock = threading.Lock()
        self._minute: dict[str, deque] = defaultdict(deque)
        self._day: dict[str, deque] = defaultdict(deque)
        self._checks_since_sweep = 0

    def _evict_idle(self, now: float) -> None:
        """Drop clients with no activity inside the daily window.

        Called under the lock. Cheap because it only runs every N checks, and
        the daily deque is the authority on whether a client is still relevant.
        """
        for key in [k for k, stamps in self._day.items() if not stamps or now - stamps[-1] > 86_400]:
            self._day.pop(key, None)
            self._minute.pop(key, None)

        # Hard ceiling in case a burst of distinct keys arrives faster than the
        # daily window retires them: drop the least recently seen.
        if len(self._day) > self._MAX_TRACKED_CLIENTS:
            ordered = sorted(self._day.items(), key=lambda kv: kv[1][-1] if kv[1] else 0)
            for key, _ in ordered[: len(self._day) - self._MAX_TRACKED_CLIENTS]:
                self._day.pop(key, None)
                self._minute.pop(key, None)

    @property
    def tracked_clients(self) -> int:
        with self._lock:
            return len(self._day)

    def check(self, client: str) -> tuple[bool, str, int]:
        """Returns (allowed, reason, retry_after_seconds)."""
        now = time.time()
        with self._lock:
            self._checks_since_sweep += 1
            if self._checks_since_sweep >= self._SWEEP_EVERY:
                self._checks_since_sweep = 0
                self._evict_idle(now)

            minute = self._minute[client]
            day = self._day[client]

            while minute and now - minute[0] > 60:
                minute.popleft()
            while day and now - day[0] > 86_400:
                day.popleft()

            if self.per_minute and len(minute) >= self.per_minute:
                return False, f"{self.per_minute} requests/minute", int(61 - (now - minute[0]))
            if self.per_day and len(day) >= self.per_day:
                return False, f"{self.per_day} requests/day", int(86_401 - (now - day[0]))

            minute.append(now)
            day.append(now)
            return True, "", 0

    def snapshot(self, client: str) -> dict:
        now = time.time()
        with self._lock:
            minute = sum(1 for t in self._minute.get(client, ()) if now - t <= 60)
            day = sum(1 for t in self._day.get(client, ()) if now - t <= 86_400)
        return {
            "used_this_minute": minute,
            "limit_per_minute": self.per_minute,
            "used_today": day,
            "limit_per_day": self.per_day,
        }


# Defaults sit just under the Groq free tier so the local limiter, not the
# upstream 429, is what a caller hits first.
limiter = SlidingWindowLimiter(
    per_minute=_env_int("RATE_LIMIT_PER_MINUTE", 20),
    per_day=_env_int("RATE_LIMIT_PER_DAY", 500),
)


class TokenBudgetLimiter:
    """Sliding-window limiter over TOKENS, not requests.

    Requests per minute is the limit that is easy to model and the wrong one to
    model. On Groq's free tier a chat model allows 30 requests/minute but only
    8,000 tokens/minute, and this application's requests are large: a system
    prompt plus extracted facts plus a structured response is several thousand
    tokens, so the token ceiling binds long before the request ceiling does.

    Counting only requests meant the app cheerfully sent a second and third
    repair attempt into a budget that was already spent, and the user got an
    upstream 429 naming a limit the application had never heard of. This tracks
    the constraint that actually binds, refuses locally before spending a round
    trip, and says how long to wait.

    Usage is recorded from the provider's own `usage.total_tokens` where
    available, and from the local estimate before the call so that concurrent
    requests cannot all pass the check on stale state.
    """

    def __init__(self, per_minute: int, per_day: int):
        self.per_minute = per_minute
        self.per_day = per_day
        self._lock = threading.Lock()
        # (timestamp, tokens) pairs.
        self._minute: dict[str, deque] = defaultdict(deque)
        self._day: dict[str, deque] = defaultdict(deque)

    def _trim(self, client: str, now: float) -> tuple[deque, deque]:
        minute, day = self._minute[client], self._day[client]
        while minute and now - minute[0][0] > 60:
            minute.popleft()
        while day and now - day[0][0] > 86_400:
            day.popleft()
        return minute, day

    def remaining(self, client: str) -> tuple[int, int]:
        now = time.time()
        with self._lock:
            minute, day = self._trim(client, now)
            used_minute = sum(t for _, t in minute)
            used_day = sum(t for _, t in day)
        return (
            max(0, self.per_minute - used_minute) if self.per_minute else 1 << 30,
            max(0, self.per_day - used_day) if self.per_day else 1 << 30,
        )

    def check(self, client: str, estimated_tokens: int) -> tuple[bool, str, int]:
        """Would a request of this size fit? Returns (allowed, reason, retry_after)."""
        now = time.time()
        with self._lock:
            minute, day = self._trim(client, now)
            used_minute = sum(t for _, t in minute)
            used_day = sum(t for _, t in day)

            if self.per_minute and used_minute + estimated_tokens > self.per_minute:
                oldest = minute[0][0] if minute else now
                return (
                    False,
                    f"{used_minute:,} of {self.per_minute:,} tokens/minute already used; "
                    f"this request needs about {estimated_tokens:,} more",
                    max(1, int(61 - (now - oldest))),
                )
            if self.per_day and used_day + estimated_tokens > self.per_day:
                oldest = day[0][0] if day else now
                return (
                    False,
                    f"{used_day:,} of {self.per_day:,} tokens/day already used; "
                    f"this request needs about {estimated_tokens:,} more",
                    max(1, int(86_401 - (now - oldest))),
                )
        return True, "", 0

    def record(self, client: str, tokens: int) -> None:
        """Charge `tokens` to the window with no intention of correcting it."""
        if tokens <= 0:
            return
        now = time.time()
        with self._lock:
            self._minute[client].append([now, tokens])
            self._day[client].append([now, tokens])

    def reserve(self, client: str, tokens: int) -> list | None:
        """Claim `tokens` now, returning a handle to settle against the truth.

        The claim has to happen before the call, not after: two requests that
        both check a stale window both pass, and the provider then refuses the
        second with a limit the application never saw. So the estimate is
        charged up front and corrected when the real figure arrives.
        """
        if tokens <= 0:
            return None
        now = time.time()
        entry = [now, tokens]
        with self._lock:
            self._minute[client].append(entry)
            self._day[client].append(entry)
        return entry

    def settle(self, handle: list | None, actual_tokens: int) -> None:
        """Replace a reservation with what the call actually cost.

        A reservation is a deliberate over-estimate: the full output allowance
        is claimed because the real output length is unknowable until the
        response arrives. Leaving that over-estimate in the window burns
        allowance nobody spent -- on an 8,000-token minute, reserving 2,600 for
        an answer that ran to 1,200 quietly costs the next call its place, and
        a seven-part analysis loses most of a call's worth of budget to
        arithmetic that was never true.

        An entry already trimmed out of the window is left alone: it has
        expired, and reviving it would charge a minute that has passed.
        """
        if handle is None:
            return
        with self._lock:
            handle[1] = max(0, actual_tokens)

    def wait_needed(self, client: str, estimated_tokens: int) -> float:
        """Seconds until a request of this size would fit. 0 if it fits now.

        Returns -1 when it can never fit, because the request alone exceeds a
        whole window -- waiting would not help and the caller must be told to
        shrink the request instead.
        """
        if self.per_minute and estimated_tokens > self.per_minute:
            return -1.0
        now = time.time()
        with self._lock:
            minute, _ = self._trim(client, now)
            used = sum(t for _, t in minute)
            if not self.per_minute or used + estimated_tokens <= self.per_minute:
                return 0.0
            # Retiring entries from the front of the window frees their tokens.
            freed = 0
            for timestamp, tokens in minute:
                freed += tokens
                if used - freed + estimated_tokens <= self.per_minute:
                    return max(0.0, 60.0 - (now - timestamp)) + 0.5
        return 60.0

    def snapshot(self, client: str) -> dict:
        remaining_minute, remaining_day = self.remaining(client)
        return {
            "limit_per_minute": self.per_minute,
            "remaining_this_minute": remaining_minute,
            "limit_per_day": self.per_day,
            "remaining_today": remaining_day,
        }


# Matched to the Groq free tier's chat-model ceiling (8,000 TPM / 200,000 TPD
# for gpt-oss-120b). Raise these if your account has a higher allowance --
# they exist to fail fast locally, not to be conservative for its own sake.
token_limiter = TokenBudgetLimiter(
    per_minute=_env_int("TOKEN_LIMIT_PER_MINUTE", 8_000),
    per_day=_env_int("TOKEN_LIMIT_PER_DAY", 200_000),
)

# Waiting for budget beats failing. A free-tier ceiling means a multi-part
# analysis genuinely takes minutes, and a result that arrives slowly is worth
# more than a 429 that arrives quickly. Bounded so a request cannot hang
# forever, and every wait is reported.
WAIT_FOR_TOKEN_BUDGET = _env_bool("WAIT_FOR_TOKEN_BUDGET", True)
MAX_TOTAL_WAIT_SECONDS = _env_int("MAX_TOTAL_WAIT_SECONDS", 600)


async def await_token_budget(
    client: str,
    estimated_tokens: int,
    *,
    stage: str,
    waited_so_far: float = 0.0,
) -> tuple[float, list | None]:
    """Block until `estimated_tokens` fit, or raise.

    Returns (seconds waited, reservation handle). The handle must be settled
    with the call's real token usage; see `TokenBudgetLimiter.settle`.

    Every path reserves. The non-waiting path used to check and return without
    charging anything, on the assumption that the caller would record the usage
    afterwards -- but the caller only recorded the difference against a
    reservation that was never made, so with WAIT_FOR_TOKEN_BUDGET off the
    limiter accumulated nothing and let every repair through. A limiter that
    silently stops limiting is worse than no limiter, because the refusal it
    was there to produce now comes from the provider instead.
    """
    if not WAIT_FOR_TOKEN_BUDGET:
        enforce_token_budget(client, estimated_tokens, stage=stage)
        return 0.0, token_limiter.reserve(client, estimated_tokens)

    wait = token_limiter.wait_needed(client, estimated_tokens)

    if wait < 0:
        raise GovernanceError(
            f"A single call for {stage} needs about {estimated_tokens:,} tokens, more than the "
            f"entire {token_limiter.per_minute:,}-token minute allowance. Waiting cannot help.",
            status=413,
            hint=(
                "Paste a smaller excerpt, or use a model with a higher token allowance. "
                "Raise TOKEN_LIMIT_PER_MINUTE if your account permits more than this "
                "deployment assumes."
            ),
        )

    if wait == 0:
        return 0.0, token_limiter.reserve(client, estimated_tokens)

    if waited_so_far + wait > MAX_TOTAL_WAIT_SECONDS:
        raise GovernanceError(
            f"Completing this request would need another {wait:.0f}s of waiting for token "
            f"budget, past the {MAX_TOTAL_WAIT_SECONDS}s limit for one request "
            f"({waited_so_far:.0f}s already spent waiting).",
            status=429,
            hint=(
                "The analysis was split into more parts than this minute's allowance can "
                "carry. Narrow the input, raise MAX_TOTAL_WAIT_SECONDS, or use a model with "
                "a higher token allowance."
            ),
            headers={"Retry-After": str(int(wait))},
        )

    await asyncio.sleep(wait)
    # Reserve immediately so concurrent requests cannot both claim the freed
    # budget; the real usage replaces this reservation once the call returns.
    return wait, token_limiter.reserve(client, estimated_tokens)


def _parse_limit_header(value: str | None) -> int | None:
    """Parse a rate-limit header value like '8000' or '200000'."""
    if not value:
        return None
    try:
        return int(float(value.strip()))
    except (TypeError, ValueError):
        return None


def adopt_provider_limits(headers) -> dict | None:
    """Tune the local token budget from the provider's own rate-limit headers.

    Limits differ per model -- on Groq's free tier gpt-oss-120b allows 8,000
    tokens/minute while prompt-guard allows 15,000, and a day's allowance
    differs too. Any number compiled into this application is therefore a guess
    about someone else's account.

    OpenAI-compatible providers report the real values on every response
    (`x-ratelimit-limit-tokens`, `x-ratelimit-remaining-tokens`). Adopting them
    makes the local budget track the account actually in use, which is the same
    principle applied to quota that the model registry applies to context
    windows. Headers absent, configured defaults stand.
    """
    limit = _parse_limit_header(headers.get("x-ratelimit-limit-tokens"))
    remaining = _parse_limit_header(headers.get("x-ratelimit-remaining-tokens"))
    if limit is None or limit <= 0:
        return None

    changed = limit != token_limiter.per_minute
    token_limiter.per_minute = limit

    day_limit = _parse_limit_header(headers.get("x-ratelimit-limit-tokens-day"))
    if day_limit and day_limit > 0:
        token_limiter.per_day = day_limit

    return {
        "limit_per_minute": limit,
        "provider_remaining": remaining,
        "adopted": changed,
    }


def enforce_token_budget(client: str, estimated_tokens: int, *, stage: str = "this request") -> None:
    allowed, reason, retry_after = token_limiter.check(client, estimated_tokens)
    if not allowed:
        raise GovernanceError(
            f"Token budget exceeded before {stage}: {reason}.",
            status=429,
            hint=(
                f"Free-tier accounts are limited by tokens per minute, not requests. "
                f"Wait {retry_after}s and retry, use a smaller input, or raise "
                f"TOKEN_LIMIT_PER_MINUTE if your account allows more. This was refused "
                f"locally, so no quota was spent upstream."
            ),
            headers={"Retry-After": str(retry_after)},
        )

# Bounds how many upstream calls are in flight at once. Without it, twenty
# concurrent browser tabs become twenty simultaneous Groq calls and a 429.
_MAX_CONCURRENCY = _env_int("MAX_CONCURRENT_REQUESTS", 4)
_semaphore: asyncio.Semaphore | None = None


def concurrency_slot() -> asyncio.Semaphore:
    """Lazily created so the semaphore binds to the running event loop."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
    return _semaphore


def enforce_rate_limit(client: str) -> None:
    allowed, reason, retry_after = limiter.check(client)
    if not allowed:
        raise GovernanceError(
            f"Rate limit exceeded: {reason}.",
            status=429,
            hint=(
                "This is the application's own limit, set below the upstream free-tier quota "
                "so that a burst is shed here rather than consuming your daily allowance. "
                "Tune with RATE_LIMIT_PER_MINUTE and RATE_LIMIT_PER_DAY."
            ),
            headers={"Retry-After": str(max(1, retry_after))},
        )


# --- authentication -------------------------------------------------------


@dataclass
class AuthPolicy:
    """Optional shared-secret auth.

    Off by default, because requiring a token to use a tool bound to localhost
    is friction with no benefit. It exists so that binding to a routable
    interface does not mean running open, and startup warns when it is not on.
    """

    tokens: list[str] = field(default_factory=lambda: _env_list("API_TOKENS"))

    @property
    def enabled(self) -> bool:
        return bool(self.tokens)

    def check(self, header_value: str | None) -> None:
        if not self.enabled:
            return
        presented = (header_value or "").removeprefix("Bearer ").strip()
        # Constant-time compare so a token cannot be recovered by timing.
        if not any(hmac.compare_digest(presented, token) for token in self.tokens):
            raise GovernanceError(
                "Missing or invalid API token.",
                status=401,
                hint="Send 'Authorization: Bearer <token>' with a value from API_TOKENS.",
            )

    def public(self) -> dict:
        return {"auth_required": self.enabled, "configured_tokens": len(self.tokens)}


model_policy = ModelPolicy()
auth_policy = AuthPolicy()


# --- CORS -----------------------------------------------------------------


def cors_origins() -> list[str]:
    """Explicit origins, defaulting to loopback rather than a wildcard.

    `*` is convenient and is what the earlier version shipped. It is also how a
    page on any origin gets to call a service bound to a shared interface, so
    the default is the two loopback origins and anything wider is opt-in.
    """
    configured = _env_list("CORS_ORIGINS")
    if configured:
        return configured
    port = os.getenv("PORT", "8000")
    return [f"http://127.0.0.1:{port}", f"http://localhost:{port}"]


def deployment_warnings(host: str) -> list[str]:
    """Checks worth shouting about at startup."""
    warnings: list[str] = []
    exposed = host not in {"127.0.0.1", "localhost", "::1"}

    if exposed and not auth_policy.enabled:
        warnings.append(
            f"Bound to {host} (not loopback) with no authentication. Anyone who can reach this "
            f"port can spend your API quota and submit data to the upstream provider. "
            f"Set API_TOKENS, or bind to 127.0.0.1."
        )
    if exposed and "*" in cors_origins():
        warnings.append("CORS_ORIGINS includes '*' on a non-loopback bind. Restrict it to known origins.")
    if exposed and _env_bool("ENABLE_TEST_EXECUTION", True):
        warnings.append(
            "Test execution is enabled on a non-loopback bind. Generated code from any caller "
            "will be executed on this host. Set ENABLE_TEST_EXECUTION=false or run in a container."
        )
    return warnings
