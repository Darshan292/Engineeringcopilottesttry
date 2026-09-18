"""Request/response models for the tool endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# Guardrail against someone pasting a 10MB file into a textarea and burning
# their whole daily rate-limit budget on one doomed request.
MAX_INPUT_CHARS = 60_000


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
                f"limit. Paste a smaller excerpt."
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
    markdown: str
    model: str
    finish_reason: str | None = None
    elapsed_ms: int
    usage: Usage
    truncated: bool = Field(
        default=False,
        description="True when the model hit its output cap and the answer is cut short.",
    )


class ErrorResponse(BaseModel):
    error: str
    hint: str | None = None
