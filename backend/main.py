"""Engineering Copilot -- FastAPI app entrypoint.

Serves the API under /api and the static frontend at /. No database, no auth,
no persistence: every request is independent and nothing is stored between
them. Run with `python -m backend.main` or `uvicorn backend.main:app --reload`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import settings
from .governance import cors_origins, deployment_warnings
from .routes.tools import router as tools_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("copilot")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(
    title="Engineering Copilot",
    description=(
        "Four local engineering tools backed by OpenRouter's free tier: unit test "
        "generation, API docs, log/RCA summaries, and blameless postmortems. "
        "Stateless -- nothing is persisted between requests."
    ),
    version="0.1.0",
)

# Loopback by default rather than a wildcard: `*` is what lets a page on any
# origin call this service the moment it is bound to a routable interface.
# Widen deliberately with CORS_ORIGINS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*", "Authorization"],
    expose_headers=["X-Request-ID", "Retry-After"],
)

app.include_router(tools_router)


@app.exception_handler(RequestValidationError)
async def validation_handler(_request: Request, exc: RequestValidationError):
    """Turn pydantic's nested error blobs into one readable sentence."""
    first = exc.errors()[0] if exc.errors() else {}
    message = str(first.get("msg", "Invalid request.")).removeprefix("Value error, ")
    return JSONResponse(
        status_code=422,
        content={"error": message, "hint": "Check the pasted input and try again."},
    )


@app.get("/api/health", tags=["meta"])
async def health():
    """Liveness plus a readiness signal for the one thing that can be misconfigured."""
    return {
        "status": "ok",
        "api_key_configured": settings.has_api_key,
        "model": settings.model,
        "model_warning": settings.model_warning,
    }


async def _announce_billing_posture(provider) -> None:
    """Load the provider's price list and say plainly what may be spent."""
    from .billing import free_tier_only, guard
    from .llm_client import list_models

    if not free_tier_only():
        log.warning(
            "  billing: FREE_TIER_ONLY is off -- paid models may be called and CHARGED."
        )
    else:
        log.info("  billing: free models only; anything with a non-zero price is refused")

    if not settings.has_api_key:
        return

    try:
        await list_models()  # populates the guard as a side effect
    except Exception as exc:
        log.warning("  billing: could not read %s's price list (%s).", provider.label, exc)
        if free_tier_only():
            log.warning(
                "  billing: calls are BLOCKED until it can be read, because no model can be "
                "confirmed free. This is deliberate."
            )
        return

    free = guard.free_models()
    log.info("  billing: %d free model(s) available to this key", len(free))

    configured = (settings.model or "").strip()
    if not configured:
        return
    try:
        guard.assert_free(configured)
    except Exception as exc:
        log.warning("  billing: '%s' is NOT callable -- %s", configured, exc)
        if free:
            log.warning("  billing: free alternatives include %s", ", ".join(free[:5]))
    else:
        log.info("  billing: '%s' is confirmed free", configured)


@app.on_event("startup")
async def announce() -> None:
    from .providers import api_key_env_name, detect_provider

    provider = detect_provider(settings.base_url)
    log.info("Engineering Copilot starting")
    log.info("  provider: %s (%s)", provider.label, settings.base_url)
    log.info("  model: %s", settings.model or "(not set)")
    if not settings.has_api_key:
        key_name = api_key_env_name(provider)
        log.warning("  %s is NOT set -- every tool call will fail with 503.", key_name)
        log.warning(
            "  Fix: cp .env.example .env, then paste a key from %s",
            provider.console_keys_url or "your provider's console",
        )
    if settings.model_warning:
        log.warning("  %s", settings.model_warning)

    # Read the price list before serving anything. The guard refuses every call
    # until this succeeds, so doing it now turns "the first request failed with
    # a confusing 503" into a line in the startup log that says what to fix.
    await _announce_billing_posture(provider)

    for warning in deployment_warnings(settings.host):
        log.warning("  SECURITY: %s", warning)

    log.info("  CORS origins: %s", ", ".join(cors_origins()))
    log.info(
        "  test execution: %s",
        "enabled (generated code runs in a subprocess on this host)"
        if settings.test_execution_enabled
        else "disabled",
    )
    log.info("  UI:   http://%s:%s/", settings.host, settings.port)
    log.info("  Docs: http://%s:%s/docs", settings.host, settings.port)


# Static frontend last, so /api/* always wins over the catch-all mount.
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        # No icon file shipped. A bare 204 keeps the browser from logging a 404;
        # it must have no body at all, so this cannot be a JSONResponse.
        return Response(status_code=204)
else:  # pragma: no cover - only if someone deletes frontend/
    log.warning("frontend/ not found at %s -- API only.", FRONTEND_DIR)


def run() -> None:
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=bool(__debug__),
    )


if __name__ == "__main__":
    run()
