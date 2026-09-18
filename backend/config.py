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
DEFAULT_MODEL = "qwen/qwen3.6-27b"

KNOWN_FREE_TIER_MODELS: tuple[dict[str, str], ...] = (
    {
        "id": "qwen/qwen3.6-27b",
        "note": "Default. Groq's recommended replacement for llama-3.3-70b-versatile. ~131K context.",
    },
    {
        "id": "openai/gpt-oss-120b",
        "note": "Larger open-weight MoE, also on the free tier. ~131K context, bigger output budget.",
    },
    {
        "id": "openai/gpt-oss-20b",
        "note": "Smaller and faster sibling. Good when you want snappier turnaround.",
    },
)

RETIRED_MODELS: dict[str, str] = {
    "llama-3.3-70b-versatile": "Decommissioned on the Groq free/developer tier on 2026-08-16.",
    "llama-3.1-8b-instant": "Deprecated alongside llama-3.3-70b-versatile on 2026-06-17.",
    "mixtral-8x7b-32768": "Long retired.",
    "llama3-70b-8192": "Long retired.",
    "llama3-8b-8192": "Long retired.",
}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


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
