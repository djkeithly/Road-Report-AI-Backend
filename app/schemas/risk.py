"""Risk prediction request/response schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RiskTier = Literal["very-low", "low", "moderate", "high", "severe"]


class RiskRequest(BaseModel):
    """Request body for risk prediction."""

    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    road_name: str | None = Field(default=None, description="Road name if known")
    segment: str | None = Field(default=None, description="Road segment description")
    weather_condition: str | None = Field(
        default=None,
        description="Override weather condition value",
    )
    road_class: str | None = Field(default=None, description="Road class label")
    query_time_iso: datetime | None = Field(
        default=None,
        description="Optional query time in ISO format",
    )


class RiskCoordinates(BaseModel):
    """Coordinate object used by frontend consumers."""

    lat: float
    lng: float


class RiskDetail(BaseModel):
    """A line item contributing to a component score."""

    label: str
    value: str


class RiskComponent(BaseModel):
    """Weighted score component with explainability details."""

    name: str
    key: Literal["C", "A", "E", "T"]
    score: float = Field(..., ge=0, le=100)
    max_points: int = Field(default=25, alias="maxPoints")
    weight: float = Field(..., ge=0, le=1)
    details: list[RiskDetail]
    source: str
    model_config = ConfigDict(populate_by_name=True)


class RiskComponents(BaseModel):
    """Four-part risk scoring structure."""

    road_condition: RiskComponent = Field(alias="roadCondition")
    historical: RiskComponent
    environmental: RiskComponent
    traffic: RiskComponent
    model_config = ConfigDict(populate_by_name=True)


class WeatherSnapshot(BaseModel):
    """Observed or forecast weather inputs used for scoring."""

    short_forecast: str | None = Field(default=None, alias="shortForecast")
    temperature_f: int | None = Field(default=None, alias="temperatureF")
    wind_speed: str | None = Field(default=None, alias="windSpeed")
    source: str
    model_config = ConfigDict(populate_by_name=True)


class RiskResponse(BaseModel):
    """Response with crash risk score and explainability data."""

    risk_score: float = Field(..., ge=0, le=1, description="Crash likelihood score 0-1")
    total: int = Field(..., ge=0, le=100, description="Display score 0-100")
    tier: RiskTier
    road: str
    segment: str
    latitude: float
    longitude: float
    coordinates: RiskCoordinates
    updated_at: datetime = Field(alias="updatedAt")
    components: RiskComponents
    summary: str
    advice: str
    weather: WeatherSnapshot
    warnings: list[str] = Field(default_factory=list)
    message: str = "Risk score generated"
    model_config = ConfigDict(populate_by_name=True)
