"""FastAPI application setup."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import models  # noqa: F401
from app.config import get_settings
from app.database import Base, engine
from app.api.routes import health, risk

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create DB tables. Shutdown: dispose engine."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(
    title="Road Report AI API",
    description="Crash risk prediction API for road safety insights",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for Vue.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health.router, prefix=settings.api_v1_prefix)
app.include_router(risk.router, prefix=settings.api_v1_prefix)


@app.get("/")
def root() -> dict[str, str]:
    """Return API info and links to documentation."""
    return {
        "message": "Road Report AI Backend",
        "docs": "/docs",
        "api": settings.api_v1_prefix,
    }


def _error_payload(
    *,
    code: str,
    message: str,
    details: dict | list | str | None = None,
) -> dict[str, dict[str, str | dict | list]]:
    """Return standard SSoT error envelope."""
    error: dict[str, str | dict | list] = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    return {"error": error}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return standardized validation error envelope."""
    return JSONResponse(
        status_code=422,
        content=_error_payload(
            code="VALIDATION_ERROR",
            message="Request validation failed.",
            details=exc.errors(),
        ),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """Return standardized HTTP error envelope."""
    code_map = {
        401: "UNAUTHORIZED",
        404: "NOT_FOUND",
        429: "RATE_LIMITED",
    }
    error_code = code_map.get(exc.status_code, "INTERAL_ERROR")
    message = (
        "Internal server error."
        if exc.status_code >= 500
        else str(exc.detail or "Request failed.")
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(code=error_code, message=message),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Return standardized internal error envelope for uncaught exceptions."""
    return JSONResponse(
        status_code=500,
        content=_error_payload(
            code="INTERAL_ERROR",
            message="Internal server error.",
        ),
    )
