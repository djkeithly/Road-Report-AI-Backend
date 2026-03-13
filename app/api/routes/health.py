"""Health check endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.schemas.health import ModelMetadataResponse
from app.services.risk import get_model_runtime_metadata

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)) -> dict[str, str]:
    """Check API and database connectivity."""
    await db.execute(text("SELECT 1"))
    return {"status": "ok", "message": "Database connected and API is healthy"}


@router.get("/health/model", response_model=ModelMetadataResponse)
async def model_health(db: AsyncSession = Depends(get_db)) -> ModelMetadataResponse:
    """Report model artifact readiness and latest training metrics."""
    await db.execute(text("SELECT 1"))
    metadata = get_model_runtime_metadata()
    if not metadata["available"]:
        return {
            "status": "degraded",
            "message": str(metadata["message"]),
            "model_path": str(metadata["model_path"]),
            "available": False,
        }

    return {
        "status": "ok",
        "message": "Model artifact and metadata loaded.",
        "model_path": str(metadata["model_path"]),
        "available": True,
        "input_size": int(metadata["input_size"]),
        "rows_used": int(metadata["rows_used"]),
        "threshold": float(metadata["threshold"]),
        "accuracy": float(metadata["accuracy"]),
        "precision": float(metadata["precision"]),
        "recall": float(metadata["recall"]),
        "f1": float(metadata["f1"]),
    }
