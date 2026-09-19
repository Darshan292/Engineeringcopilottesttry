"""Refuse to spend money, structurally rather than by intention.

The operator's requirement is absolute: this deployment must never incur a
charge. "Point it at a free model and be careful" does not satisfy that, because
every way it goes wrong is quiet. A `:free` suffix can be dropped in a typo. A
free variant can be withdrawn and the paid one answer in its place. A router can
dispatch to whatever it likes. An account with credits spends them without ever
returning an error, so the first symptom is a bill.

So the rule is inverted. Nothing is callable until it has been shown to be free,
and "shown" means the provider's own price list says every rate is zero -- not
that the id looks right.

Three independent layers, because any one of them can be wrong:

1. **Before the call.** The model must appear in the provider's live catalogue
   with every pricing field at zero. A model that is absent, unpriced, or priced
   at anything above zero is refused. Unknown means refused, not assumed free.
2. **In the request.** OpenRouter is told the ceiling it may spend, so the
   refusal happens at their end too, on the request this process never saw.
3. **After the call.** Responses report what they cost. A non-zero figure means
   layers 1 and 2 were both wrong, so the guard latches shut and every later
   call in the process is refused until someone restarts it having understood
   why.

The third layer is the one that matters most, because it is the only one that
can catch a mistake this module has not thought of.

`FREE_TIER_ONLY=false` turns all of it off, for someone deliberately spending
money. It defaults to on, and the startup log says which.
"""

from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger("copilot.billing")

# Every rate a provider can quote. Checking only prompt and completion is how a
# model with free tokens and a per-request fee slips through; OpenRouter prices
# requests, images, reasoning and web search separately.
PRICING_FIELDS = (
    "prompt",
    "completion",
    "request",
    "image",
    "audio",
    "web_search",
    "internal_reasoning",
    "input_cache_read",
    "input_cache_write",
)

# Routers that select across the whole catalogue, paid models included. Their
# own price list is empty or nominal, so the pricing check cannot see the cost
# and they have to be named. `openrouter/free` is deliberately absent: it routes
# only within free models, which is the entire point of it.
ALWAYS_PAID_MODELS = frozenset(
    {
        "openrouter/auto",
        "openrouter/default",
    }
)


def free_tier_only() -> bool:
    """Whether to enforce. On unless explicitly switched off."""
    raw = (os.getenv("FREE_TIER_ONLY") or "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


class BillingRefused(Exception):
    """A call was stopped because it might cost money.

    Carries the same `status`/`hint` shape as GroqError so the route layer can
    surface it without a special case.
    """

    def __init__(self, message: str, *, hint: str | None = None, status: int = 402):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.status = status


def _rate_is_zero(value) -> bool:
    """True only when this rate is definitely zero.

    Absent is fine -- providers omit what does not apply. Unparseable is not:
    a rate this code cannot read is a rate it cannot verify, and the whole
    point is to fail closed.
    """
    if value is None or value == "":
        return True
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def price_check(model_entry: dict | None) -> tuple[bool, str]:
    """Is this catalogue entry free? Returns (free, why-not)."""
    if not isinstance(model_entry, dict):
        return False, "it is not in the provider's catalogue"

    pricing = model_entry.get("pricing")
    if not isinstance(pricing, dict) or not pricing:
        return False, "the provider quotes no prices for it, so it cannot be shown to be free"

    charged = []
    unreadable = []
    for name in PRICING_FIELDS:
        if name not in pricing:
            continue
        value = pricing[name]
        if _rate_is_zero(value):
            continue
        try:
            float(value)
        except (TypeError, ValueError):
            unreadable.append(f"{name}={value!r}")
        else:
            charged.append(f"{name}={value}")

    if unreadable:
        return False, f"its price list could not be read ({', '.join(unreadable)})"
    if charged:
        return False, f"it charges {', '.join(charged)}"
    return True, ""


class FreeTierGuard:
    """Process-wide enforcement, with a latch that cannot be cleared at runtime."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # model id -> (is_free, reason)
        self._verdicts: dict[str, tuple[bool, str]] = {}
        self._catalogue_loaded = False
        self._prices_published = False
        # Set once a real charge is observed. Deliberately not resettable: the
        # only safe response to "we were charged" is to stop and be looked at.
        self._tripped: str | None = None
        self._observed_cost = 0.0

    # --- catalogue ---------------------------------------------------------

    def load_catalogue(self, models: list[dict]) -> int:
        """Record a verdict per model from the provider's live price list."""
        verdicts: dict[str, tuple[bool, str]] = {}
        priced = 0
        for entry in models or []:
            model_id = str(entry.get("id") or "").strip()
            if not model_id:
                continue
            if isinstance(entry.get("pricing"), dict) and entry["pricing"]:
                priced += 1
            if model_id in ALWAYS_PAID_MODELS:
                verdicts[model_id] = (False, "it is a router that may select paid models")
                continue
            verdicts[model_id] = price_check(entry)
        with self._lock:
            self._verdicts = verdicts
            self._catalogue_loaded = True
            # Not every provider publishes prices. Groq's /models lists a
            # context window and nothing else, because on Groq "free tier" is a
            # property of the account rather than of the model -- there is no
            # per-model rate to check and nothing this guard can verify.
            # Pretending to enforce there would be worse than not enforcing:
            # it would refuse every model, and the operator would switch the
            # guard off entirely and lose it on the provider that needs it.
            self._prices_published = priced > 0
        free = sum(1 for ok, _ in verdicts.values() if ok)
        if self._prices_published:
            log.info("billing guard: %d of %d catalogue models are free", free, len(verdicts))
        else:
            log.info(
                "billing guard: this provider publishes no per-model prices, so the price "
                "check cannot apply; the cost tripwire on responses still does"
            )
        return free

    @property
    def prices_published(self) -> bool:
        with self._lock:
            return self._prices_published

    @property
    def catalogue_loaded(self) -> bool:
        with self._lock:
            return self._catalogue_loaded

    def free_models(self) -> list[str]:
        with self._lock:
            return sorted(mid for mid, (ok, _) in self._verdicts.items() if ok)

    # --- the three layers --------------------------------------------------

    def assert_not_tripped(self) -> None:
        if self._tripped:
            raise BillingRefused(
                f"Refusing to call the model: this process already recorded a charge "
                f"({self._tripped}). Every request is blocked until it is restarted.",
                hint=(
                    "This should be impossible on a free model, so treat it as real. Check "
                    "your OpenRouter activity page before restarting, and confirm LLM_MODEL "
                    "names a model whose every rate is zero."
                ),
            )

    def assert_free(self, model: str) -> None:
        """Layer 1. Refuse anything not positively shown to be free."""
        if not free_tier_only():
            return
        self.assert_not_tripped()

        model_id = (model or "").strip()
        if not model_id:
            raise BillingRefused("No model configured, so nothing can be checked as free.")

        if model_id in ALWAYS_PAID_MODELS:
            raise BillingRefused(
                f"'{model_id}' selects from the whole catalogue, including paid models.",
                hint=(
                    "Use 'openrouter/free', which routes only within free models, or name a "
                    "specific free model. See GET /api/models."
                ),
            )

        with self._lock:
            loaded = self._catalogue_loaded
            verdict = self._verdicts.get(model_id)

        if loaded and not self._prices_published:
            # Nothing to check against. The response-cost tripwire still runs,
            # and `public()` reports this rather than implying a guarantee that
            # is not being made.
            return

        if not loaded:
            # Better to stop than to guess. The catalogue is one cheap GET and
            # the startup hook fetches it; arriving here means that failed.
            raise BillingRefused(
                "The provider's price list could not be read, so no model can be confirmed "
                "free. Refusing to call anything rather than risk a charge.",
                hint=(
                    "Check network access to the provider and restart. To proceed without "
                    "this protection, set FREE_TIER_ONLY=false -- which accepts that calls "
                    "may be billed."
                ),
                status=503,
            )

        if verdict is None:
            raise BillingRefused(
                f"'{model_id}' is not in the provider's catalogue, so it cannot be confirmed "
                f"free.",
                hint=(
                    f"Check the id against GET /api/models. Free ids usually end in ':free'. "
                    f"A withdrawn free variant is exactly how a paid model ends up being "
                    f"called by accident."
                ),
                status=404,
            )

        is_free, reason = verdict
        if not is_free:
            raise BillingRefused(
                f"Refusing to call '{model_id}': {reason}.",
                hint=(
                    "This deployment is configured to make only free calls "
                    "(FREE_TIER_ONLY=true). Pick a model listed as free by GET /api/models, "
                    "or set FREE_TIER_ONLY=false to allow paid ones."
                ),
            )

    def request_guard_fields(self, provider_name: str) -> dict:
        """Layer 2. Body fields that make the provider enforce this too.

        OpenRouter's `provider.max_price` caps the price per million tokens it
        is willing to route to, and fails the request when every candidate
        provider is above it. Zero therefore means "only somewhere that charges
        nothing", enforced at their end on the call this process cannot see.

        That covers the gap layer 1 cannot: a catalogue that went stale between
        the check and the call, and a router whose choice is made after we have
        stopped looking.
        """
        if not free_tier_only() or provider_name != "openrouter":
            return {}
        return {"provider": {"max_price": {"prompt": 0, "completion": 0}}}

    def observe_usage(self, model: str, usage: dict | None) -> float:
        """Layer 3. Believe the invoice over every assumption above it."""
        if not isinstance(usage, dict):
            return 0.0
        raw = usage.get("cost")
        if raw is None:
            cost_details = usage.get("cost_details")
            if isinstance(cost_details, dict):
                raw = cost_details.get("upstream_inference_cost")
        if raw is None:
            return 0.0
        try:
            cost = float(raw)
        except (TypeError, ValueError):
            return 0.0

        if cost <= 0:
            return 0.0

        with self._lock:
            self._observed_cost += cost
        log.error("billing guard: '%s' reported a cost of %s", model, cost)

        if free_tier_only() and self._tripped is None:
            self._tripped = (
                f"'{model}' reported a cost of {cost} on a call this deployment had "
                f"verified as free"
            )
        return cost

    # --- reporting ---------------------------------------------------------

    def public(self) -> dict:
        enforced = free_tier_only()
        return {
            "enforced": enforced,
            # What is actually being guaranteed, rather than what was asked for.
            "mode": (
                "off"
                if not enforced
                else "price-verified"
                if self.prices_published
                else "cost-tripwire-only (provider publishes no per-model prices)"
            ),
            "catalogue_loaded": self.catalogue_loaded,
            "free_models_known": len(self.free_models()),
            "observed_cost": round(self._observed_cost, 6),
            "blocked": bool(self._tripped),
            "blocked_reason": self._tripped,
        }


guard = FreeTierGuard()
