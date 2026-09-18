"""Request/response models for the tool endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# A backstop against a genuinely absurd paste, not a capability limit. Input
# size is governed downstream by the token budget and the chunk planner, which
# compress what they can and refuse the rest with a specific reason and remedy.
# The old 60k cap predated that layer and rejected logs the system now handles.
MAX_INPUT_CHARS = 5_000_000


class ToolRequest(BaseModel):
    input: str = Field(..., description="Raw text pasted by the user.")
    model: str | None = Field(
        default=None,
        description="Optional per-request model override. Falls back to GROQ_MODEL.",
    )
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)

    @field_validator("input")
    @classmethod
    def _non_empty_and_bounded(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Input is empty. Paste some content first.")
        if len(stripped) > MAX_INPUT_CHARS:
            raise ValueError(
                f"Input is {len(stripped):,} characters, over the {MAX_INPUT_CHARS:,} "
                f"character limit. Large inputs below that limit are handled by "
                f"compression and chunking, so narrow the excerpt to the window you "
                f"care about rather than splitting it arbitrarily."
            )
        return stripped

    @field_validator("model")
    @classmethod
    def _clean_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class Usage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ToolResponse(BaseModel):
    tool: str
    request_id: str = ""
    markdown: str
    model: str
    elapsed_ms: int
    usage: dict = Field(default_factory=dict)
    # Surfaced prominently in the UI: redactions performed, injection attempts
    # neutralized, input-kind mismatches, context compression.
    warnings: list[str] = Field(default_factory=list)
    # Per-stage machine-readable results: parse stats, grounding ratios,
    # computed confidence factors, test-execution outcome, OpenAPI validation.
    diagnostics: dict = Field(default_factory=dict)
    # How many model calls it took to get a valid response, and why the
    # rejected ones were rejected.
    attempts: int = 1
    repairs: list[str] = Field(default_factory=list)
    trace: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    hint: str | None = None
    request_id: str = ""
