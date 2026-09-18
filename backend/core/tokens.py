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


def context_window_for(model: str) -> int:
    if model in CONTEXT_WINDOWS:
        return CONTEXT_WINDOWS[model]
    # Match on the bare name when the operator uses a provider prefix we do
    # not have listed, e.g. "groq/qwen/qwen3.6-27b".
    tail = model.rsplit("/", 1)[-1]
    for known, window in CONTEXT_WINDOWS.items():
        if known.rsplit("/", 1)[-1] == tail:
            return window
    return DEFAULT_CONTEXT_WINDOW


@dataclass
class TokenBudget:
    """How much input a request can carry, after everything else is paid for."""

    model: str
    context_window: int
    reserved_for_output: int
    reserved_for_system: int
    safety_margin: int
    available_for_input: int
    calibration: dict = field(default_factory=dict)

    def fits(self, estimated_input_tokens: int) -> bool:
        return estimated_input_tokens <= self.available_for_input

    def public(self) -> dict:
        return {
            "model": self.model,
            "context_window": self.context_window,
            "reserved_for_output": self.reserved_for_output,
            "reserved_for_system": self.reserved_for_system,
            "safety_margin": self.safety_margin,
            "available_for_input": self.available_for_input,
            "calibration": self.calibration,
        }


def build_budget(
    model: str,
    system_prompt: str,
    max_output_tokens: int,
    *,
    context_window: int | None = None,
) -> TokenBudget:
    """Compute the usable input budget for one request."""
    window = context_window or context_window_for(model)
    cal = calibrator.for_model(model)

    system_tokens = cal.apply(estimate_tokens(system_prompt)) + 16

    # The margin covers estimator error. It starts wide and narrows once the
    # calibrator has seen enough real completions to be trusted.
    margin_pct = 0.08 if cal.confident else 0.18
    margin = max(256, int(window * margin_pct))

    available = window - max_output_tokens - system_tokens - margin
    return TokenBudget(
        model=model,
        context_window=window,
        reserved_for_output=max_output_tokens,
        reserved_for_system=system_tokens,
        safety_margin=margin,
        available_for_input=max(0, available),
        calibration=cal.public(),
    )
