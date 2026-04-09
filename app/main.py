"""FastAPI application setup."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
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
