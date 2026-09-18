"""Tool endpoints.

Thin on purpose. Governance, then the pipeline, then a response. Everything
interesting happens in `pipeline/tools.py`; this layer's only jobs are turning
policy violations into the right HTTP status and making sure every response
carries the request ID and the diagnostics needed to audit it.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from ..config import settings
from ..governance import (
    GovernanceError,
    auth_policy,
    concurrency_slot,
    enforce_rate_limit,
    limiter,
    model_policy,
)
from ..groq_client import GroqError, list_models
from ..pipeline.tools import PIPELINES
from ..samples import SAMPLES
from ..schemas import ErrorResponse, ToolRequest, ToolResponse

log = logging.getLogger("copilot.api")
router = APIRouter(prefix="/api", tags=["tools"])

TOOL_META: dict[str, dict[str, str]] = {
    "unit-tests": {
        "title": "Unit Test Generator",
        "blurb": "AST-extracted boundary conditions, tests written against them, then actually executed.",
        "placeholder": "Paste a function, class, or module here...",
    },
    "api-docs": {
        "title": "API Doc Generator",
        "blurb": "Routes and error paths parsed from source; the OpenAPI is schema-validated and cross-checked.",
        "placeholder": "Paste your route/endpoint/controller code here...",
    },
    "log-rca": {
        "title": "Log / RCA Summarizer",
        "blurb": "Logs parsed and clustered; every claim cites a line ID that is verified to exist.",
        "placeholder": "Paste a log excerpt here...",
    },
    "postmortem": {
        "title": "Postmortem Drafter",
        "blurb": "Names are stripped before the model sees anything. Blameless by construction, not by request.",
        "placeholder": "Paste the incident channel transcript here...",
    },
}


def _client_key(request: Request) -> str:
    """Identify the caller for rate limiting.

    Client host, not a forwarded header: `X-Forwarded-For` is caller-controlled
    and trusting it turns the limiter into a decoration. A real deployment
    behind a proxy should set the limit at the proxy.
    """
    return request.client.host if request.client else "unknown"


def _error(message: str, status: int, hint: str | None, request_id: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(error=message, hint=hint, request_id=request_id).model_dump(),
        headers={"X-Request-ID": request_id, **(headers or {})},
    )


async def _handle(tool: str, request: Request, body: ToolRequest, authorization: str | None) -> JSONResponse:
    request_id = uuid.uuid4().hex[:12]

    try:
        auth_policy.check(authorization)
        model = model_policy.check_model(body.model, settings.groq_model)
        temperature = model_policy.check_temperature(body.temperature)
        enforce_rate_limit(_client_key(request))
    except GovernanceError as exc:
        log.warning("rid=%s tool=%s rejected: %s", request_id, tool, exc.message)
        return _error(exc.message, exc.status, exc.hint, request_id, exc.headers)

    log.info("rid=%s tool=%s chars=%d model=%s", request_id, tool, len(body.input), model)

    try:
        # Bound in-flight upstream calls so a burst of tabs cannot turn into a
        # burst of simultaneous provider requests.
        async with concurrency_slot():
            result = await PIPELINES[tool](
                body.input, model=model, temperature=temperature, request_id=request_id
            )
    except GroqError as exc:
        log.warning("rid=%s tool=%s failed: %s", request_id, tool, exc.message)
        return _error(exc.message, exc.status, exc.hint, request_id)
    except Exception as exc:  # pragma: no cover - last-resort guard
        log.exception("rid=%s tool=%s unhandled", request_id, tool)
        return _error(
            f"Unexpected error in the {tool} pipeline: {type(exc).__name__}",
            500,
            "This is a bug. The request ID above appears in the server log next to the traceback.",
            request_id,
        )

    log.info(
        "rid=%s tool=%s ok attempts=%d tokens=%s ms=%d",
        request_id, tool, result.attempts, result.usage.get("total_tokens"), result.elapsed_ms,
    )

    response = ToolResponse(
        tool=tool,
        request_id=request_id,
        markdown=result.markdown,
        model=result.model,
        elapsed_ms=result.elapsed_ms,
        usage=result.usage,
        warnings=result.warnings,
        diagnostics=result.diagnostics,
        attempts=result.attempts,
        repairs=result.repairs,
        trace=result.trace.public(),
    )
    return JSONResponse(content=response.model_dump(), headers={"X-Request-ID": request_id})


# --- the four tools -------------------------------------------------------


@router.post("/unit-tests", response_model=ToolResponse, responses={429: {"model": ErrorResponse}})
async def unit_tests(request: Request, body: ToolRequest, authorization: str | None = Header(default=None)):
    """Extract structure from pasted code, generate tests, and execute them."""
    return await _handle("unit-tests", request, body, authorization)


@router.post("/api-docs", response_model=ToolResponse, responses={429: {"model": ErrorResponse}})
async def api_docs(request: Request, body: ToolRequest, authorization: str | None = Header(default=None)):
    """Extract routes and error paths, then produce validated OpenAPI plus prose."""
    return await _handle("api-docs", request, body, authorization)


@router.post("/log-rca", response_model=ToolResponse, responses={429: {"model": ErrorResponse}})
async def log_rca(request: Request, body: ToolRequest, authorization: str | None = Header(default=None)):
    """Parse and cluster logs, then produce an evidence-verified incident brief."""
    return await _handle("log-rca", request, body, authorization)


@router.post("/postmortem", response_model=ToolResponse, responses={429: {"model": ErrorResponse}})
async def postmortem(request: Request, body: ToolRequest, authorization: str | None = Header(default=None)):
    """Anonymize a transcript, then draft a blameless postmortem from it."""
    return await _handle("postmortem", request, body, authorization)


# --- introspection --------------------------------------------------------


@router.get("/tools")
async def tools():
    """Everything the frontend needs to render its tabs, including samples."""
    return {
        "tools": [
            {"id": tool_id, **meta, "sample": SAMPLES.get(tool_id, {})}
            for tool_id, meta in TOOL_META.items()
        ]
    }


@router.get("/config")
async def config(request: Request):
    """Non-secret runtime config, plus this caller's current rate-limit usage."""
    return {
        **settings.public_dict(),
        "governance": {**model_policy.public(), **auth_policy.public()},
        "rate_limit": limiter.snapshot(_client_key(request)),
    }


@router.get("/models")
async def models():
    """Live model list from Groq. The only trustworthy source -- model IDs churn."""
    try:
        return {"models": await list_models(), "configured": settings.groq_model}
    except GroqError as exc:
        return _error(exc.message, exc.status, exc.hint, "n/a")


@router.get("/samples/{tool_id}")
async def sample(tool_id: str):
    """One realistic sample input for a tool."""
    if tool_id not in SAMPLES:
        raise HTTPException(status_code=404, detail=f"No sample for tool '{tool_id}'.")
    return SAMPLES[tool_id]
