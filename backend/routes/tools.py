"""The four tool endpoints, plus config/model introspection.

Every tool is the same shape -- tailored system prompt + raw user text -> Groq
-> markdown -- so they share one handler rather than four near-identical
copies. Adding a fifth tool means adding a prompt and a sample, nothing here.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..config import settings
from ..groq_client import GroqError, complete, list_models
from ..prompts import TOOL_PROMPTS, USER_FRAMING
from ..samples import SAMPLES
from ..schemas import ErrorResponse, ToolRequest, ToolResponse

router = APIRouter(prefix="/api", tags=["tools"])

TOOL_META: dict[str, dict[str, str]] = {
    "unit-tests": {
        "title": "Unit Test Generator",
        "blurb": "Paste a function or class. Get a runnable test suite plus the edge cases you forgot.",
        "placeholder": "Paste a function, class, or module here...",
    },
    "api-docs": {
        "title": "API Doc Generator",
        "blurb": "Paste route handlers. Get OpenAPI 3.1 YAML and a human-readable reference.",
        "placeholder": "Paste your route/endpoint/controller code here...",
    },
    "log-rca": {
        "title": "Log / RCA Summarizer",
        "blurb": "Paste a log excerpt. Get an incident brief with a timeline and a confidence-rated root cause.",
        "placeholder": "Paste a log excerpt here...",
    },
    "postmortem": {
        "title": "Postmortem Drafter",
        "blurb": "Paste an incident chat transcript. Get a blameless postmortem with role placeholders, not names.",
        "placeholder": "Paste the incident channel transcript here...",
    },
}


def _error(exc: GroqError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status,
        content=ErrorResponse(error=exc.message, hint=exc.hint).model_dump(),
    )


async def _run_tool(tool: str, request: ToolRequest) -> ToolResponse | JSONResponse:
    system_prompt = TOOL_PROMPTS[tool]
    user_content = USER_FRAMING[tool].format(input=request.input)

    try:
        result = await complete(
            system_prompt,
            user_content,
            model=request.model,
            temperature=request.temperature,
        )
    except GroqError as exc:
        return _error(exc)

    return ToolResponse(
        tool=tool,
        markdown=result["text"],
        model=result["model"],
        finish_reason=result["finish_reason"],
        elapsed_ms=result["elapsed_ms"],
        usage=result["usage"],
        truncated=result["finish_reason"] == "length",
    )


# --- The four tools -------------------------------------------------------


@router.post("/unit-tests", response_model=ToolResponse)
async def unit_tests(request: ToolRequest):
    """Generate a test suite from pasted source code."""
    return await _run_tool("unit-tests", request)


@router.post("/api-docs", response_model=ToolResponse)
async def api_docs(request: ToolRequest):
    """Generate OpenAPI YAML and prose docs from pasted route code."""
    return await _run_tool("api-docs", request)


@router.post("/log-rca", response_model=ToolResponse)
async def log_rca(request: ToolRequest):
    """Summarize a log excerpt into an incident brief with a root-cause hypothesis."""
    return await _run_tool("log-rca", request)


@router.post("/postmortem", response_model=ToolResponse)
async def postmortem(request: ToolRequest):
    """Draft a blameless postmortem from an incident chat transcript."""
    return await _run_tool("postmortem", request)


# --- Introspection --------------------------------------------------------


@router.get("/tools")
async def tools():
    """Everything the frontend needs to render its tabs, including samples."""
    return {
        "tools": [
            {
                "id": tool_id,
                **meta,
                "sample": SAMPLES.get(tool_id, {}),
            }
            for tool_id, meta in TOOL_META.items()
        ]
    }


@router.get("/config")
async def config():
    """Non-secret runtime config, so the UI can show the active model and warn on bad setup."""
    return settings.public_dict()


@router.get("/models")
async def models():
    """Live model list from Groq. The only trustworthy source -- model IDs churn."""
    try:
        return {"models": await list_models(), "configured": settings.groq_model}
    except GroqError as exc:
        return _error(exc)


@router.get("/samples/{tool_id}")
async def sample(tool_id: str):
    """One realistic sample input for a tool."""
    if tool_id not in SAMPLES:
        raise HTTPException(status_code=404, detail=f"No sample for tool '{tool_id}'.")
    return SAMPLES[tool_id]
