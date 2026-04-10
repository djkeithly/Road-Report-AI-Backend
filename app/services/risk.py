"""Risk prediction service with weather enrichment and scoring scaffold."""

import json
import os
import pickle
from datetime import UTC, datetime
from functools import lru_cache
from math import fabs
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from app.config import get_settings
from app.schemas.risk import (
    RiskComponent,
    RiskComponents,
    RiskCoordinates,
    RiskDetail,
    RiskRequest,
    RiskResponse,
    RiskTier,
)
from app.services.weather import get_fallback_weather_snapshot, get_weather_snapshot


# 1. Define the Model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def _tier_from_score(score_100: int) -> RiskTier:
    """Map display score to a frontend tier label."""
    if score_100 <= 20:
        return "very-low"
    if score_100 <= 40:
        return "low"
    if score_100 <= 60:
        return "moderate"
    if score_100 <= 80:
        return "high"
    return "severe"


def _build_components(
    *,
    latitude: float,
    longitude: float,
    road_class: str | None,
    weather_condition: str | None,
    temperature_f: int | None,
) -> tuple[RiskComponents, list[str]]:
    """Assemble explainable weighted components and warnings."""
    warnings: list[str] = []

    road_condition_score = 48.0 + (abs(latitude) % 7)
    historical_score = 42.0 + (abs(longitude) % 9)

    environmental_score = 40.0
    if weather_condition:
        condition = weather_condition.lower()
        if "rain" in condition or "storm" in condition:
            environmental_score = 68.0
        if "snow" in condition or "ice" in condition:
            environmental_score = 79.0
    elif temperature_f is None:
        warnings.append(
            "Weather data unavailable; environmental component estimated conservatively."
        )

    traffic_score = 45.0 + (fabs(latitude - longitude) % 8)

    components = RiskComponents(
        roadCondition=RiskComponent(
            name="Road Condition",
            key="C",
            score=round(road_condition_score, 2),
            maxPoints=25,
            weight=0.30,
            details=[
                RiskDetail(
                    label="Road Class", value=requested_value_or_dash(road_class)
                ),
                RiskDetail(label="Surface Condition", value="Unknown"),
            ],
            source="txdot-cris-estimate",
        ),
        historical=RiskComponent(
            name="Historical Crash Pattern",
            key="A",
            score=round(historical_score, 2),
            maxPoints=25,
            weight=0.30,
            details=[
                RiskDetail(label="Crash trend window", value="30-day baseline"),
                RiskDetail(label="County profile", value="Estimated"),
            ],
            source="txdot-cris-estimate",
        ),
        environmental=RiskComponent(
            name="Environmental Conditions",
            key="E",
            score=round(environmental_score, 2),
            maxPoints=25,
            weight=0.25,
            details=[
                RiskDetail(
                    label="Forecast condition", value=weather_condition or "Unavailable"
                ),
                RiskDetail(
                    label="Temperature (F)",
                    value=(
                        str(temperature_f)
                        if temperature_f is not None
                        else "Unavailable"
                    ),
                ),
            ],
            source="weather.gov",
        ),
        traffic=RiskComponent(
            name="Traffic Pattern",
            key="T",
            score=round(traffic_score, 2),
            maxPoints=25,
            weight=0.15,
            details=[
                RiskDetail(label="Time profile", value="Current local window"),
                RiskDetail(label="Congestion proxy", value="Estimated"),
            ],
            source="model-estimate",
        ),
    )
    return components, warnings


def _infer_road_class_from_name(road_name: str | None) -> str | None:
    """Infer coarse road class from road name patterns."""
    if not road_name:
        return None
    text = road_name.strip().lower()
    if not text:
        return None
    if text.startswith(("i-", "ih", "interstate")):
        return "INTERSTATE"
    if text.startswith(("us ", "us-", "us/")):
        return "US & STATE HIGHWAYS"
    if text.startswith(("sh", "state highway", "hwy", "highway")):
        return "US & STATE HIGHWAYS"
    if "toll" in text or "turnpike" in text or "tpke" in text:
        return "TOLLWAY"
    if text.startswith("fm") or "farm to market" in text:
        return "FARM TO MARKET"
    return "CITY STREET"


def requested_value_or_dash(value: str | None) -> str:
    """Return fallback marker used by frontend details cards."""
    if not value:
        return "-"
    return value


def _score_from_components(components: RiskComponents) -> float:
    """Calculate normalized 0-1 risk score from weighted components."""
    component_scores = [
        components.road_condition.score * components.road_condition.weight,
        components.historical.score * components.historical.weight,
        components.environmental.score * components.environmental.weight,
        components.traffic.score * components.traffic.weight,
    ]
    score_100 = sum(component_scores)
    normalized = max(0.0, min(1.0, score_100 / 100))
    return normalized


@lru_cache(maxsize=1)
def _load_model_bundle() -> dict[str, object] | None:
    """Load trained Logistic Regression model, feature columns, and banding cutoffs."""
    model_path = Path("logistic_regression_model.pth")
    cols_path = Path("feature_columns.pkl")

    if (
        not model_path.exists()
        and (Path(__file__).resolve().parents[2] / model_path).exists()
    ):
        model_path = Path(__file__).resolve().parents[2] / model_path
        cols_path = Path(__file__).resolve().parents[2] / cols_path

    if not model_path.exists() or not cols_path.exists():
        return None

    # Load the feature columns mapping
    with open(cols_path, "rb") as f:
        feature_columns = pickle.load(f)

    # Initialize and load the model weights
    input_dim = len(feature_columns)
    model = LogisticRegression(input_dim)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    )
    model.eval()

    # Pre-calculate banding cutoffs using the reference dataset
    banding_dataset = [
        {
            "City": ["DALLAS"],
            "County": ["DALLAS"],
            "Crash Month": ["1"],
            "Crash Time": ["0"],
            "Rural Urban Type": ["LARGE URBANIZED (200,000+)"],
            "Street Name": ["S I 35E S"],
            "Surface Condition": ["1 - DRY"],
            "Weather Condition": ["1 - CLEAR"],
        },
        {
            "City": ["DALLAS"],
            "County": ["DALLAS"],
            "Crash Month": ["1"],
            "Crash Time": ["0"],
            "Rural Urban Type": ["No Data"],
            "Street Name": ["S I 35E S"],
            "Surface Condition": ["1 - DRY"],
            "Weather Condition": ["1 - CLEAR"],
        },
        {
            "City": ["DALLAS"],
            "County": ["DALLAS"],
            "Crash Month": ["1"],
            "Crash Time": ["0"],
            "Rural Urban Type": ["No Data"],
            "Street Name": ["S I 35E S"],
            "Surface Condition": ["2 - WET"],
            "Weather Condition": ["3 - RAIN"],
        },
    ]

    banding_cutoffs = []
    for band_item in banding_dataset:
        custom_df = pd.DataFrame(band_item)
        custom_encoded = pd.get_dummies(custom_df)
        custom_aligned = custom_encoded.reindex(columns=feature_columns, fill_value=0)
        custom_tensor = torch.tensor(custom_aligned.to_numpy(dtype=np.float32))

        with torch.no_grad():
            prob = torch.sigmoid(model(custom_tensor)).item()
            banding_cutoffs.append(prob)

    return {
        "model": model,
        "feature_columns": feature_columns,
        "banding_cutoffs": banding_cutoffs,
    }


async def _get_city_from_coords(lat: float, lon: float) -> str:
    """Reverse geocode coordinates to find the city using OpenStreetMap Nominatim."""
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10"
    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": "CrashRiskApp/1.0"}
        ) as client:
            response = await client.get(url, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                address = data.get("address", {})
                city = (
                    address.get("city") or address.get("town") or address.get("village")
                )
                if city:
                    return city.upper()
    except Exception:
        pass

    return "UNKNOWN"


def _build_inference_df(
    *,
    request: RiskRequest,
    weather_condition: str | None,
    query_time: datetime,
    city: str,
) -> pd.DataFrame:
    """Build a single-row Pandas DataFrame compatible with dummy encoding."""
    weather_upper = (weather_condition or "CLEAR").upper()
    if "RAIN" in weather_upper or "STORM" in weather_upper:
        w_cond = "3 - RAIN"
        s_cond = "2 - WET"
    elif "SNOW" in weather_upper or "ICE" in weather_upper:
        w_cond = "4 - SNOW"
        s_cond = "3 - ICE"
    else:
        w_cond = "1 - CLEAR"
        s_cond = "1 - DRY"

    crash_time_formatted = str(query_time.hour * 100)
    crash_month_formatted = str(query_time.month)

    segment_text = (request.segment or "").lower()
    if "downtown" in segment_text:
        rural_urban_type = "LARGE URBANIZED (200,000+)"
    else:
        rural_urban_type = "No Data"

    row = {
        "City": [city],
        "County": ["DALLAS"],
        "Crash Month": [crash_month_formatted],
        "Crash Time": [crash_time_formatted],
        "Rural Urban Type": [rural_urban_type],
        "Street Name": [request.road_name.upper() if request.road_name else "UNKNOWN"],
        "Surface Condition": [s_cond],
        "Weather Condition": [w_cond],
    }
    return pd.DataFrame(row)


def _infer_model_probability(
    *,
    request: RiskRequest,
    weather_condition: str | None,
    query_time: datetime,
    city: str,
) -> tuple[float | None, list[float] | None]:
    """Return model probability and dynamically calculated banding cutoffs."""
    bundle = _load_model_bundle()
    if bundle is None:
        return None, None

    df = _build_inference_df(
        request=request,
        weather_condition=weather_condition,
        query_time=query_time,
        city=city,
    )

    encoded = pd.get_dummies(df)
    aligned = encoded.reindex(columns=bundle["feature_columns"], fill_value=0)
    tensor = torch.tensor(aligned.to_numpy(dtype=np.float32))

    model = bundle["model"]
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor)).item()

    return probability, bundle["banding_cutoffs"]


def get_model_runtime_metadata() -> dict[str, str | int | float | bool]:
    """Return model artifact metadata for runtime diagnostics and UI display."""
    model_path = Path("logistic_regression_model.pth")
    if (
        not model_path.exists()
        and (Path(__file__).resolve().parents[2] / model_path).exists()
    ):
        model_path = Path(__file__).resolve().parents[2] / model_path

    if not model_path.exists():
        return {
            "available": False,
            "model_path": str(model_path),
            "message": "Logistic Regression model artifact not found.",
        }

    return {
        "available": True,
        "model_path": str(model_path),
        "message": "Using PyTorch Logistic Regression model",
    }


async def predict_risk(request: RiskRequest) -> RiskResponse:
    """Predict crash risk for a location using banded Logistic Regression."""
    warnings: list[str] = []
    weather = get_fallback_weather_snapshot()
    try:
        weather = await get_weather_snapshot(request.latitude, request.longitude)
    except (httpx.HTTPError, KeyError, ValueError):
        warnings.append(
            "Live weather source unavailable; using fallback environmental data."
        )

    query_time = request.query_time_iso or datetime.now(tz=UTC)
    weather_condition = request.weather_condition or weather.short_forecast
    inferred_road_class = request.road_class or _infer_road_class_from_name(
        request.road_name
    )

    components, component_warnings = _build_components(
        latitude=request.latitude,
        longitude=request.longitude,
        road_class=inferred_road_class,
        weather_condition=weather_condition,
        temperature_f=weather.temperature_f,
    )
    warnings.extend(component_warnings)

    city = await _get_city_from_coords(request.latitude, request.longitude)
    heuristic_score = _score_from_components(components)
    model_prob, cutoffs = _infer_model_probability(
        request=request,
        weather_condition=weather_condition,
        query_time=query_time,
        city=city,
    )

    if model_prob is not None and cutoffs is not None:
        c0, c1, c2 = cutoffs[0], cutoffs[1], cutoffs[2]

        # Piecewise interpolation to guarantee score ranges per band
        if model_prob <= c0:
            band_index = 0
            # Scale probability to 0.00 - 0.09 (Single digits)
            risk_score = (model_prob / max(1e-6, c0)) * 0.09

        elif model_prob <= c1:
            band_index = 1
            # Scale probability to 0.10 - 0.49
            risk_score = 0.10 + ((model_prob - c0) / max(1e-6, c1 - c0)) * 0.39

        elif model_prob <= c2:
            band_index = 2
            # Scale probability to 0.50 - 0.89
            risk_score = 0.50 + ((model_prob - c1) / max(1e-6, c2 - c1)) * 0.39

        else:
            band_index = 3
            # Scale probability to 0.90 - 1.00 (Above 90)
            risk_score = 0.90 + ((model_prob - c2) / max(1e-6, 1.0 - c2)) * 0.10

        risk_score = max(0.0, min(1.0, risk_score))

        warnings.append(
            f"Logistic Regression Probability: {model_prob:.4f} (Band {band_index})."
        )
    else:
        warnings.append(
            "Model artifact unavailable; using deterministic fallback scoring."
        )
        risk_score = heuristic_score

    score_100 = int(round(risk_score * 100))
    tier = _tier_from_score(score_100)

    road = request.road_name or "Unknown road segment"
    segment = request.segment or "Segment details unavailable"

    summary = (
        f"Logistic Regression model estimates a {score_100}% crash risk profile for "
        f"the selected location."
    )
    advice = (
        "Increase caution, reduce speed, and monitor nearby incidents when risk is high. "
        "Use local advisories before travel decisions."
    )

    return RiskResponse(
        risk_score=risk_score,
        total=score_100,
        tier=tier,
        road=road,
        segment=segment,
        latitude=request.latitude,
        longitude=request.longitude,
        coordinates=RiskCoordinates(lat=request.latitude, lng=request.longitude),
        updatedAt=query_time,
        components=components,
        summary=summary,
        advice=advice,
        weather=weather,
        warnings=warnings,
        message="Risk score generated",
    )
