"""Health and model metadata response schemas."""

from typing import Literal

from pydantic import BaseModel


class ModelMetadataResponse(BaseModel):
    """Model metadata response for health and frontend metrics endpoints."""

    status: Literal["ok", "degraded"]
    message: str
    model_path: str
    feature_columns_path: str | None = None
    model_type: str | None = None
    model_version: str | None = None
    available: bool | None = None
    input_size: int | None = None
    known_road_count: int | None = None
    rows_used: int | None = None
    threshold: float | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
