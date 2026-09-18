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
        "Four local engineering tools backed by the Groq free tier: unit test "
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
        "model": settings.groq_model,
        "model_warning": settings.model_warning,
    }


@app.on_event("startup")
async def announce() -> None:
    log.info("Engineering Copilot starting")
    log.info("  model: %s", settings.groq_model)
    if not settings.has_api_key:
        log.warning("  GROQ_API_KEY is NOT set -- every tool call will fail with 503.")
        log.warning("  Fix: cp .env.example .env, then paste a key from https://console.groq.com/keys")
    if settings.model_warning:
        log.warning("  %s", settings.model_warning)

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
