"""Health check endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.schemas.health import ModelMetadataResponse
from app.services.risk import get_model_runtime_metadata

router = APIRouter(tags=["health"])


def _metadata_payload(metadata: dict[str, str | int | float | bool]) -> dict:
    """Normalize model metadata dictionary to schema-compatible response payload."""
    return {
        "status": "ok",
        "message": "Model artifact and metadata loaded.",
        "model_path": str(metadata.get("model_path", "")),
        "feature_columns_path": str(metadata.get("feature_columns_path", "")),
        "model_type": str(metadata.get("model_type", "")),
        "model_version": str(metadata.get("model_version", "")),
        "available": True,
        "input_size": (
            int(metadata["input_size"]) if metadata.get("input_size") is not None else None
        ),
        "known_road_count": (
            int(metadata["known_road_count"])
            if metadata.get("known_road_count") is not None
            else None
        ),
        "rows_used": (
            int(metadata["rows_used"]) if metadata.get("rows_used") is not None else None
        ),
        "threshold": (
            float(metadata["threshold"]) if metadata.get("threshold") is not None else None
        ),
        "accuracy": (
            float(metadata["accuracy"]) if metadata.get("accuracy") is not None else None
        ),
        "precision": (
            float(metadata["precision"]) if metadata.get("precision") is not None else None
        ),
        "recall": float(metadata["recall"]) if metadata.get("recall") is not None else None,
        "f1": float(metadata["f1"]) if metadata.get("f1") is not None else None,
    }


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
            "feature_columns_path": str(metadata.get("feature_columns_path", "")),
            "available": False,
        }

    return _metadata_payload(metadata)
