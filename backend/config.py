"""Environment-driven configuration.

Everything is read once at import time so the rest of the app can treat
settings as plain attributes. No secrets are ever logged or returned to the
browser -- `public_dict()` is the only thing the frontend sees.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .providers import detect_provider, resolve_api_key, resolve_base_url
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root if present. Real environment variables win.
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env", override=False)


# ---------------------------------------------------------------------------
# Model defaults
#
# There is deliberately no default model.
#
# OpenRouter's catalogue is hundreds of models deep, and the free variants are
# the most volatile part of it: providers donate capacity and withdraw it, and a
# `:free` suffix that resolves today can 404 next week. Any id compiled in here
# would be a guess that eventually becomes a confusing failure for someone who
# never chose it. Being asked to pick beats being told a model you never picked
# is gone, so the app points at the live list instead -- which is also the list
# the billing guard checks prices against.
# ---------------------------------------------------------------------------
DEFAULT_MODELS: dict[str, str] = {
    "openrouter": "",
    "generic": "",
}

# Model ids that are known to be gone, so the failure names the cause rather
# than surfacing a bare 404. Populated from experience rather than compiled
# from a vendor list, because the live catalogue is the source of truth.
RETIRED_MODELS: dict[str, str] = {}

# Models that are a router in front of other models rather than a model
# themselves. They matter for budgeting: the rate limit that binds is the
# underlying model's, a 429 names a model the caller never chose, and the prompt
# on the wire carries routing instructions the text we sent gives no hint of --
# so a token estimate built from our own messages understates the real cost.
#
# `openrouter/free` is here because it dispatches across free models. It is
# still safe to use: it only ever selects from models priced at zero, which is
# exactly what the billing guard requires. `openrouter/auto` is deliberately
# NOT here -- it is blocked outright in `billing.py`, because it selects from
# the entire catalogue and no price check can see its choice in advance.
ROUTING_MODELS: frozenset[str] = frozenset({"openrouter/free"})


def is_routing_model(model: str) -> bool:
    return (model or "").strip().lower() in ROUTING_MODELS


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# How many output tokens each tool actually needs. A flat reservation is
# wasteful where it is too big and truncating where it is too small, and on a
# hard tokens-per-minute ceiling the waste is the difference between a request
# fitting and being refused: reserving 4,096 for a reply that runs to 1,800
# spends a quarter of an 8,000-token minute on nothing.
#
# Sized from what each schema actually produces. `api-docs` is the outlier
# because a full OpenAPI document is long.
TOOL_OUTPUT_TOKENS: dict[str, int] = {
    "unit-tests": 2_600,
    "api-docs": 3_600,
    "log-rca": 2_200,
    "postmortem": 2_800,
}

# A map-reduce partial is a narrower answer over a slice of the input, so it
# needs far less room than the final combined document.
MAP_OUTPUT_TOKENS = 1_200


def output_tokens_for(tool: str, *, is_map_step: bool = False) -> int:
    if is_map_step:
        return MAP_OUTPUT_TOKENS
    configured = os.getenv("LLM_MAX_TOKENS")
    if configured and configured.strip().isdigit():
        return int(configured)
    return TOOL_OUTPUT_TOKENS.get(tool, 2_600)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve_model() -> str:
    """The model to call, under any of the accepted variable names.

    `LLM_MODEL` is the current name; `LLM_MODEL` is kept because every existing
    .env uses it. The per-provider default only applies where this application
    can name a model that is actually likely to exist -- see `DEFAULT_MODELS`.
    """
    for name in ("LLM_MODEL", "LLM_MODEL"):
        raw = os.getenv(name)
        if raw and raw.strip():
            return raw.strip()
    return DEFAULT_MODELS.get(detect_provider().name, "")


@dataclass(frozen=True)
class Settings:
    provider: str = field(default_factory=lambda: detect_provider().name)
    api_key: str = field(default_factory=lambda: resolve_api_key(detect_provider()))
    model: str = field(default_factory=_resolve_model)
    base_url: str = field(
        default_factory=lambda: resolve_base_url(detect_provider())
    )
    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.2))
    max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 4096))
    timeout_seconds: float = field(
        default_factory=lambda: _env_float("LLM_TIMEOUT_SECONDS", 120.0)
    )
    host: str = field(default_factory=lambda: os.getenv("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    @property
    def model_warning(self) -> str | None:
        """Warn loudly if the configured model is missing or known to be dead."""
        provider = detect_provider(self.base_url)
        if not self.model:
            return (
                f"No model is configured. {provider.label} has no default here because its "
                f"catalogue changes too often for any id compiled into this application to "
                f"stay true. Set LLM_MODEL in your .env -- GET /api/models lists what your key "
                f"can actually call right now, with the free ones marked."
            )
        reason = RETIRED_MODELS.get(self.model)
        if reason:
            replacement = DEFAULT_MODELS.get(provider.name)
            fix = (
                f"Set LLM_MODEL={replacement} in your .env instead."
                if replacement
                else "Pick another from GET /api/models and set LLM_MODEL in your .env."
            )
            return (
                f"LLM_MODEL is set to '{self.model}', which is retired. {reason} "
                f"Requests will fail with 404 model_not_found. {fix}"
            )
        return None

    @property
    def test_execution_enabled(self) -> bool:
        return (os.getenv("ENABLE_TEST_EXECUTION", "true") or "").strip().lower() not in {
            "false", "0", "no", "off",
        }

    def public_dict(self) -> dict:
        """Safe-to-serve view. Deliberately contains no key material."""
        return {
            "test_execution_enabled": self.test_execution_enabled,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "api_key_configured": self.has_api_key,
            "model_warning": self.model_warning,
            "provider": detect_provider(self.base_url).public(),
        }


settings = Settings()
