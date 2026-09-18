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
