"""Risk prediction endpoints."""

from typing import Annotated

from app.api.deps import get_db
from app.schemas.health import ModelMetadataResponse
from app.schemas.risk import RiskRequest, RiskResponse
from app.services.risk import (
    get_model_runtime_metadata,
    get_road_suggestions,
    predict_risk,
)
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/risk", tags=["risk"])


def _metadata_payload(metadata: dict[str, str | int | float | bool]) -> dict:
    """Normalize model metadata dictionary to schema-compatible response payload."""
    return {
        "status": "ok",
        "message": "Model metadata available.",
        "available": True,
        "model_path": str(metadata.get("model_path", "")),
        "feature_columns_path": str(metadata.get("feature_columns_path", "")),
        "model_type": str(metadata.get("model_type", "")),
        "model_version": str(metadata.get("model_version", "")),
        "input_size": (
            int(metadata["input_size"])
            if metadata.get("input_size") is not None
            else None
        ),
        "known_road_count": (
            int(metadata["known_road_count"])
            if metadata.get("known_road_count") is not None
            else None
        ),
        "rows_used": (
            int(metadata["rows_used"])
            if metadata.get("rows_used") is not None
            else None
        ),
        "threshold": (
            float(metadata["threshold"])
            if metadata.get("threshold") is not None
            else None
        ),
        "accuracy": (
            float(metadata["accuracy"])
            if metadata.get("accuracy") is not None
            else None
        ),
        "precision": (
            float(metadata["precision"])
            if metadata.get("precision") is not None
            else None
        ),
        "recall": (
            float(metadata["recall"]) if metadata.get("recall") is not None else None
        ),
        "f1": float(metadata["f1"]) if metadata.get("f1") is not None else None,
    }


@router.post("/predict", response_model=RiskResponse)
async def get_risk_score(
    request: RiskRequest, db: AsyncSession = Depends(get_db)
) -> RiskResponse:
    """
    Get crash risk score for a location.

    Accepts coordinates (and optional context) and returns a 0–1 risk score.
    Non-blocking; delegates to dedicated service for inference.
    """
    return await predict_risk(request, db)


@router.get("/model-metrics", response_model=ModelMetadataResponse)
async def get_model_metrics() -> ModelMetadataResponse:
    """Return model metadata for frontend metrics cards and debugging."""
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


@router.get("/roads/suggest")
async def suggest_roads(
    q: Annotated[str, Query(min_length=1, max_length=64)],
    limit: Annotated[int, Query(ge=1, le=25)] = 8,
) -> dict[str, list[str]]:
    """Return model-backed road autocomplete suggestions."""
    return {"suggestions": get_road_suggestions(q, limit=limit)}
