"""Risk prediction request/response schemas."""

from pydantic import BaseModel, Field


class RiskRequest(BaseModel):
    """Request body for risk prediction."""

    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    # Add more inputs as your model requires (e.g. time, weather, road type)
    # road_type: str | None = None
    # time_of_day: str | None = None
    # weather_condition: str | None = None


class RiskResponse(BaseModel):
    """Response with crash risk score."""

    risk_score: float = Field(..., ge=0, le=1, description="Crash likelihood score 0-1")
    latitude: float
    longitude: float
    message: str = "Risk score generated"
