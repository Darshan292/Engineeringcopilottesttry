"""Environment-driven configuration.

Everything is read once at import time so the rest of the app can treat
settings as plain attributes. No secrets are ever logged or returned to the
browser -- `public_dict()` is the only thing the frontend sees.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root if present. Real environment variables win.
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env", override=False)


# ---------------------------------------------------------------------------
# Model defaults
#
# Checked against Groq's published model/deprecation list at build time
# (2026-09-18). `llama-3.3-70b-versatile` -- the historical default for this
# kind of app -- was deprecated for free/developer tiers on 2026-06-17 and
# decommissioned on 2026-08-16. Requesting it now returns 404 model_not_found.
# Groq's own recommended replacements are the two listed below.
#
# These are hints for humans and for the /api/models fallback only; the app
# never hard-blocks a model name. Whatever GROQ_MODEL says is what gets sent.
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "openai/gpt-oss-120b"

KNOWN_FREE_TIER_MODELS: tuple[dict[str, str], ...] = (
    {
        "id": "openai/gpt-oss-120b",
        "note": "Default. ~131K context and the most reliable of these at following a JSON schema.",
    },
    {
        "id": "openai/gpt-oss-20b",
        "note": "Smaller and faster sibling. Good when you want snappier turnaround.",
    },
    {
        "id": "qwen/qwen3.8-27b",
        "note": "Alibaba's open-weight model on the free tier. ~131K context.",
    },
)

RETIRED_MODELS: dict[str, str] = {
    "llama-3.3-70b-versatile": "Decommissioned on the Groq free/developer tier on 2026-08-16.",
    "llama-3.1-8b-instant": "Deprecated alongside llama-3.3-70b-versatile on 2026-06-17.",
    "qwen/qwen3.6-27b": "Superseded by qwen/qwen3.8-27b and no longer served.",
    "mixtral-8x7b-32768": "Long retired.",
    "llama3-70b-8192": "Long retired.",
    "llama3-8b-8192": "Long retired.",
}

# Models that are a router or an agent in front of other models rather than a
# model themselves. They matter for budgeting: the rate limit that binds is the
# underlying model's, the 429 names a model the caller never chose, and the
# prompt on the wire carries tool schemas and internal instructions that the
# text we sent gives no hint of -- so a token estimate built from our own
# messages understates the real cost, on Groq's compound models by roughly 2x.
ROUTING_MODELS: frozenset[str] = frozenset({"groq/compound", "groq/compound-mini"})


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
    configured = os.getenv("GROQ_MAX_TOKENS")
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


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip())
    groq_model: str = field(
        default_factory=lambda: (os.getenv("GROQ_MODEL") or DEFAULT_MODEL).strip()
    )
    groq_base_url: str = field(
        default_factory=lambda: (
            os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1"
        ).rstrip("/")
    )
    temperature: float = field(default_factory=lambda: _env_float("GROQ_TEMPERATURE", 0.2))
    max_tokens: int = field(default_factory=lambda: _env_int("GROQ_MAX_TOKENS", 4096))
    timeout_seconds: float = field(
        default_factory=lambda: _env_float("GROQ_TIMEOUT_SECONDS", 120.0)
    )
    host: str = field(default_factory=lambda: os.getenv("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))

    @property
    def has_api_key(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def model_warning(self) -> str | None:
        """Warn loudly if the configured model is one we know is dead."""
        reason = RETIRED_MODELS.get(self.groq_model)
        if reason:
            return (
                f"GROQ_MODEL is set to '{self.groq_model}', which is retired. {reason} "
                f"Requests will fail with 404 model_not_found. "
                f"Set GROQ_MODEL={DEFAULT_MODEL} in your .env instead."
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
            "model": self.groq_model,
            "base_url": self.groq_base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "api_key_configured": self.has_api_key,
            "model_warning": self.model_warning,
            "known_free_tier_models": list(KNOWN_FREE_TIER_MODELS),
        }


settings = Settings()
