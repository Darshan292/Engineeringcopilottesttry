"""Token estimation and context budgeting.

Why not a real tokenizer: every option costs something this project cannot
pay. `tiktoken` downloads its BPE table from a hardcoded CDN on first use --
verified in this environment, where it fails outright -- which turns a local
tool into one with a hidden network dependency and a startup failure mode.
`transformers.AutoTokenizer` pulls hundreds of megabytes and the correct
vocabulary per model, and the vocabulary for whatever model the operator
configures is not knowable ahead of time.

So this estimates, and then corrects itself:

1. A pre-tokenizer splits text the way BPE implementations do (letter runs,
   digit runs, punctuation runs, whitespace), and each piece is costed with
   rules that reflect how BPE actually merges. This lands far closer than
   `len(text) / 4`, especially on logs and code, which are the inputs here.
2. Every real completion reports `usage.prompt_tokens`. The calibrator
   compares that against what was estimated for the same payload and keeps a
   per-model correction factor. The estimate converges on the truth for the
   model actually in use, without shipping a vocabulary for it.
3. Budgets apply a safety margin on top, and the margin shrinks as calibration
   confidence grows. An under-estimate costs a failed request; an
   over-estimate costs a little unused context. The asymmetry is deliberate.

The contract callers rely on: `estimate_tokens` never knowingly under-reports.
"""

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass, field

# Mirrors the pre-tokenization step in GPT-family BPE: contractions, then runs
# of letters / digits / punctuation / whitespace, each optionally led by one
# space (BPE attaches the leading space to the following word).
_PRETOKEN = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)"
    r"| ?[^\W\d_]+"
    r"| ?\d+"
    r"| ?[^\s\w]+"
    r"|\s+",
    re.UNICODE,
)


def _piece_cost(piece: str) -> int:
    """Estimated subtoken count for one pre-token."""
    stripped = piece.strip()
    if not stripped:
        # Runs of whitespace: a single space rides along with the next word,
        # but newlines and indentation are their own tokens.
        newlines = piece.count("\n")
        spaces = len(piece) - newlines
        return newlines + (spaces + 3) // 4 if spaces > 1 else max(newlines, 0)

    if stripped.isdigit():
        # BPE splits numbers into runs of 1-3 digits.
        return max(1, math.ceil(len(stripped) / 3))

    if not any(c.isalnum() for c in stripped):
        # Punctuation: common short runs merge ("):", "==") but long runs of
        # mixed symbols mostly do not.
        return max(1, math.ceil(len(stripped) / 2))

    # Words. Short and common ones are a single token; long or mixed-case
    # identifiers (snake_case, camelCase, hashes) fragment much more.
    length = len(stripped)
    if length <= 4:
        return 1
    has_mixed_case = not (stripped.islower() or stripped.isupper())
    has_digits = any(c.isdigit() for c in stripped)
    divisor = 2.6 if (has_mixed_case or has_digits) else 3.6
    return max(1, math.ceil(length / divisor))


def estimate_tokens(text: str) -> int:
    """Offline estimate of the token count for `text`."""
    if not text:
        return 0
    total = 0
    for match in _PRETOKEN.finditer(text):
        total += _piece_cost(match.group())
    # Never return 0 for non-empty input.
    return max(1, total)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate for a chat payload, including per-message framing overhead."""
    # Chat formats wrap each message in role/delimiter tokens. Four per message
    # plus a few for the reply primer is the long-standing OpenAI figure and is
    # close enough for every OpenAI-compatible server.
    total = 3
    for message in messages:
        total += 4
        total += estimate_tokens(str(message.get("content", "")))
        total += estimate_tokens(str(message.get("role", "")))
    return total


# --- calibration ----------------------------------------------------------


@dataclass
class ModelCalibration:
    """Running correction factor for one model."""

    samples: int = 0
    # Exponentially weighted mean of actual/estimated.
    ratio: float = 1.0
    last_error_pct: float = 0.0

    @property
    def confident(self) -> bool:
        return self.samples >= 5

    def observe(self, estimated: int, actual: int) -> None:
        if estimated <= 0 or actual <= 0:
            return
        observed = actual / estimated
        # Ignore wild outliers: they mean the payload was not what we measured
        # (a retry, a different message set), not that the estimator is wrong.
        if not 0.25 <= observed <= 4.0:
            return
        alpha = 0.4 if self.samples < 5 else 0.15
        self.ratio = observed if self.samples == 0 else (1 - alpha) * self.ratio + alpha * observed
        self.samples += 1
        self.last_error_pct = round((observed - 1.0) * 100, 1)

    def apply(self, estimated: int) -> int:
        return int(math.ceil(estimated * self.ratio))

    def public(self) -> dict:
        return {
            "samples": self.samples,
            "correction_factor": round(self.ratio, 4),
            "last_observed_error_pct": self.last_error_pct,
            "confident": self.confident,
        }


class Calibrator:
    """Process-wide, per-model calibration. Thread-safe, in-memory only."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, ModelCalibration] = {}

    def for_model(self, model: str) -> ModelCalibration:
        with self._lock:
            return self._models.setdefault(model, ModelCalibration())

    def observe(self, model: str, estimated: int, actual: int) -> None:
        with self._lock:
            self._models.setdefault(model, ModelCalibration()).observe(estimated, actual)

    def corrected(self, model: str, estimated: int) -> int:
        return self.for_model(model).apply(estimated)

    def public(self) -> dict:
        with self._lock:
            return {name: cal.public() for name, cal in self._models.items()}


calibrator = Calibrator()


# --- context windows ------------------------------------------------------

# Known context windows. Unknown models fall back to the conservative default
# rather than assuming a large window and failing at request time.
CONTEXT_WINDOWS: dict[str, int] = {
    "qwen/qwen3.6-27b": 131_072,
    "openai/gpt-oss-120b": 131_072,
    "openai/gpt-oss-20b": 131_072,
    "llama-3.3-70b-versatile": 131_072,
    "llama-3.1-8b-instant": 131_072,
}

DEFAULT_CONTEXT_WINDOW = 8_192

# The smallest window in which this application can do anything at all: the
# system prompt plus the output contract is ~1,500-2,300 tokens before any
# input, and a useful reply needs room too.
MIN_USABLE_CONTEXT = 6_000

# Share of the working window reserved for the model's reply. Structured output
# for these tools runs a few hundred to ~2,000 tokens; reserving more just
# starves the input side of the same budget.
OUTPUT_RESERVE_SHARE = 0.35

# Model families a provider lists that cannot serve chat completions. The
# /models endpoint returns everything an account can call -- classifiers,
# speech-to-text, text-to-speech, moderation -- and selecting one produces a
# confusing failure deep in the pipeline rather than an obvious one up front.
_NON_CHAT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("prompt-guard", "a prompt-injection classifier that returns a label, not chat completions"),
    ("llama-guard", "a content-safety classifier that returns a label, not chat completions"),
    ("whisper", "a speech-to-text model"),
    ("-tts", "a text-to-speech model"),
    ("playai-tts", "a text-to-speech model"),
    ("embed", "an embedding model"),
    ("moderation", "a moderation classifier"),
    ("-asr", "a speech recognition model"),
)


def non_chat_reason(model: str) -> str | None:
    """Why this model cannot serve this application, or None if it can."""
    lowered = model.lower()
    for marker, description in _NON_CHAT_PATTERNS:
        if marker in lowered:
            return description
    return None


def suggested_models() -> list[str]:
    """Chat models known to work, preferring what the provider confirmed."""
    known = [m for m in registry._windows if non_chat_reason(m) is None and registry._windows[m] >= MIN_USABLE_CONTEXT]
    if known:
        # Largest window first: more headroom means fewer compressions.
        return sorted(known, key=lambda m: -registry._windows[m])[:4]
    return ["openai/gpt-oss-120b", "qwen/qwen3.6-27b", "openai/gpt-oss-20b"]


class ModelRegistry:
    """Runtime model metadata, treated as authoritative over the static table.

    The hardcoded `CONTEXT_WINDOWS` above is a cold-start fallback and nothing
    more. Providers change context windows, retire models and add new ones, so
    a table baked into source is wrong the moment it ships -- and being wrong
    in the optimistic direction means requests that fail at the provider after
    the whole pipeline has already run.

    `GET /v1/models` reports `context_window` per model. This caches what the
    provider actually said, and `context_window_for` prefers it. The cache is
    refreshed opportunistically; a stale entry is still better than a guess,
    and an unknown model still falls back to the conservative default rather
    than assuming a large window.
    """

    def __init__(self, ttl_seconds: float = 3600.0, failure_backoff: float = 300.0) -> None:
        self._lock = threading.Lock()
        self._windows: dict[str, int] = {}
        self._fetched_at: float = 0.0
        self._attempted_at: float = 0.0
        self._ttl = ttl_seconds
        self._failure_backoff = failure_backoff

    def note_attempt(self) -> None:
        """Record that a refresh was tried, successful or not.

        Without this, an endpoint that does not serve /models -- a local
        llama.cpp build, a proxy, anything non-Groq -- means every single
        request pays a failing round trip forever, because the cache never
        fills and therefore never stops looking stale.
        """
        with self._lock:
            self._attempted_at = time.time()

    @property
    def should_refresh(self) -> bool:
        now = time.time()
        with self._lock:
            if self._windows and (now - self._fetched_at) <= self._ttl:
                return False
            # Back off after a failed attempt instead of retrying every request.
            return (now - self._attempted_at) > self._failure_backoff

    def update(self, models: list[dict]) -> int:
        """Record what the provider reported. Returns how many carried a window."""
        recorded = 0
        with self._lock:
            for entry in models:
                model_id = entry.get("id")
                window = entry.get("context_window")
                if model_id and isinstance(window, int) and window > 0:
                    self._windows[str(model_id)] = window
                    recorded += 1
            self._fetched_at = time.time()
            self._attempted_at = self._fetched_at
        return recorded

    def get(self, model: str) -> int | None:
        with self._lock:
            return self._windows.get(model)

    @property
    def is_stale(self) -> bool:
        with self._lock:
            return (time.time() - self._fetched_at) > self._ttl

    @property
    def has_data(self) -> bool:
        with self._lock:
            return bool(self._windows)

    def public(self) -> dict:
        with self._lock:
            return {
                "models_known": len(self._windows),
                "age_seconds": round(time.time() - self._fetched_at, 1) if self._fetched_at else None,
                "stale": (time.time() - self._fetched_at) > self._ttl if self._fetched_at else True,
            }


registry = ModelRegistry()


def context_window_for(model: str) -> tuple[int, str]:
    """Return (window, source). Runtime metadata wins over the static table."""
    live = registry.get(model)
    if live:
        return live, "provider"

    if model in CONTEXT_WINDOWS:
        return CONTEXT_WINDOWS[model], "static-table"

    # Match on the bare name when the operator uses a provider prefix we do
    # not have listed, e.g. "groq/qwen/qwen3.6-27b".
    tail = model.rsplit("/", 1)[-1]
    for known, window in CONTEXT_WINDOWS.items():
        if known.rsplit("/", 1)[-1] == tail:
            return window, "static-table (suffix match)"

    return DEFAULT_CONTEXT_WINDOW, "conservative default (model unknown)"


@dataclass
class TokenBudget:
    """How much input a request can carry, after everything else is paid for.

    `context_window` is the model's limit. `effective_window` is the smaller of
    that and what the account's tokens-per-minute allowance permits in a single
    call, and it is the number that matters. A 131,072-token context window on
    an 8,000 TPM account cannot be used: planning a 100,000-token call against
    the context window produces a plan that can never execute, which is how a
    perfectly reasonable input ends up refused.
    """

    model: str
    context_window: int
    reserved_for_output: int
    reserved_for_system: int
    safety_margin: int
    available_for_input: int
    effective_window: int = 0
    token_allowance_per_minute: int = 0
    calibration: dict = field(default_factory=dict)
    # Where the context window came from. A conservative default means the
    # budget is a guess, and callers surface that.
    window_source: str = "static-table"

    def fits(self, estimated_input_tokens: int) -> bool:
        return estimated_input_tokens <= self.available_for_input

    def public(self) -> dict:
        return {
            "model": self.model,
            "context_window": self.context_window,
            "effective_window": self.effective_window,
            "token_allowance_per_minute": self.token_allowance_per_minute,
            "binding_constraint": (
                "tokens-per-minute allowance"
                if self.effective_window < self.context_window
                else "model context window"
            ),
            "reserved_for_output": self.reserved_for_output,
            "reserved_for_system": self.reserved_for_system,
            "safety_margin": self.safety_margin,
            "available_for_input": self.available_for_input,
            "window_source": self.window_source,
            "calibration": self.calibration,
        }


def build_budget(
    model: str,
    system_prompt: str,
    max_output_tokens: int,
    *,
    context_window: int | None = None,
    token_allowance_per_minute: int | None = None,
) -> TokenBudget:
    """Compute the usable input budget for one request.

    `token_allowance_per_minute` is the account's TPM ceiling. A single call can
    never exceed it, so it caps the usable window regardless of how large the
    model's context is. Passing it is what stops the planner from producing
    plans the token budget will refuse.
    """
    if context_window:
        window, source = context_window, "explicit"
    else:
        window, source = context_window_for(model)

    # One call can never spend more than a minute's whole allowance.
    effective = window
    if token_allowance_per_minute and token_allowance_per_minute > 0:
        effective = min(window, token_allowance_per_minute)

    # The output reserve is subtracted from the same budget as the input, so a
    # fixed 4,096 on an 8,000-token working window spends half the allowance
    # before a single line of input is considered. Cap it at a share of what is
    # actually available; a structured response rarely needs more, and the
    # caller passes this back as `max_tokens` so the reservation is real.
    effective_output = min(max_output_tokens, max(512, int(effective * OUTPUT_RESERVE_SHARE)))

    cal = calibrator.for_model(model)
    system_tokens = cal.apply(estimate_tokens(system_prompt)) + 16

    # The margin covers estimator error. It starts wide and narrows once the
    # calibrator has seen enough real completions to be trusted. It scales with
    # the effective window, not the nominal one -- an 8k working budget does not
    # need a 23k margin.
    margin_pct = 0.08 if cal.confident else 0.18
    margin = max(256, int(effective * margin_pct))

    available = effective - effective_output - system_tokens - margin
    return TokenBudget(
        model=model,
        context_window=window,
        effective_window=effective,
        token_allowance_per_minute=token_allowance_per_minute or 0,
        reserved_for_output=effective_output,
        reserved_for_system=system_tokens,
        safety_margin=margin,
        available_for_input=max(0, available),
        calibration=cal.public(),
        window_source=source,
    )
