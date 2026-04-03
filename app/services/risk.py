"""Risk prediction service with weather enrichment and scoring scaffold."""

import json
from datetime import UTC, datetime
from functools import lru_cache
from math import fabs
from pathlib import Path

import httpx
import numpy as np
import torch

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
from ml.model import BaselineCrashRiskModel, load_model_state
from ml.preprocessing import build_inference_feature_vector


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
                RiskDetail(label="Road Class", value=requested_value_or_dash(road_class)),
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
                RiskDetail(label="Forecast condition", value=weather_condition or "Unavailable"),
                RiskDetail(
                    label="Temperature (F)",
                    value=str(temperature_f) if temperature_f is not None else "Unavailable",
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
    """Load trained model and metadata used for online inference."""
    settings = get_settings()
    model_path = Path(settings.model_file_path)
    if not model_path.is_absolute():
        # Resolve relative model paths from backend project root.
        model_path = Path(__file__).resolve().parents[2] / model_path
    metadata_path = model_path.with_suffix(".meta.json")
    if not model_path.exists() or not metadata_path.exists():
        return None

    with metadata_path.open("r", encoding="utf-8") as infile:
        metadata = json.load(infile)

    input_size = int(metadata["input_size"])
    feature_columns = list(metadata["feature_columns"])
    threshold = float(metadata.get("threshold", 0.5))

    model = BaselineCrashRiskModel(input_size=input_size)
    model = load_model_state(model, model_path=str(model_path))
    model.eval()

    return {
        "model": model,
        "feature_columns": feature_columns,
        "threshold": threshold,
    }


def _build_inference_row(
    *,
    request: RiskRequest,
    weather_condition: str | None,
    query_time: datetime,
) -> dict[str, str | int]:
    """Build a single-row feature payload compatible with training preprocessing."""
    hour_text = f"{query_time.hour:02d}:00 - {query_time.hour:02d}:59"
    return {
        "city": "Unknown",
        "county": "Unknown",
        "crashmonth": query_time.month,
        "crashtime": f"{query_time.hour:02d}:{query_time.minute:02d}",
        "crashyear": query_time.year,
        "dayofweek": query_time.strftime("%A").upper(),
        "hourofday": hour_text,
        "roadclass": requested_value_or_dash(request.road_class),
        "ruralurbantype": "Unknown",
        "surfacecondition": "Unknown",
        "weathercondition": weather_condition or "Unknown",
    }


def _infer_model_probability(
    *,
    request: RiskRequest,
    weather_condition: str | None,
    query_time: datetime,
) -> tuple[float | None, float | None]:
    """Return model probability and threshold when model artifacts are available."""
    bundle = _load_model_bundle()
    if bundle is None:
        return None, None

    row = _build_inference_row(
        request=request,
        weather_condition=weather_condition,
        query_time=query_time,
    )
    vector = build_inference_feature_vector(
        row=row,
        feature_columns=bundle["feature_columns"],
    )
    tensor = torch.from_numpy(np.asarray([vector], dtype=np.float32))
    model = bundle["model"]
    with torch.no_grad():
        logits = model(tensor).squeeze(-1)
        probability = float(torch.sigmoid(logits).item())
    return probability, float(bundle["threshold"])


def get_model_runtime_metadata() -> dict[str, str | int | float | bool]:
    """Return model artifact metadata for runtime diagnostics and UI display."""
    settings = get_settings()
    model_path = Path(settings.model_file_path)
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parents[2] / model_path
    metadata_path = model_path.with_suffix(".meta.json")
    if not model_path.exists() or not metadata_path.exists():
        return {
            "available": False,
            "model_path": str(model_path),
            "message": "Model artifact or metadata not found.",
        }

    with metadata_path.open("r", encoding="utf-8") as infile:
        metadata = json.load(infile)

    return {
        "available": True,
        "model_path": str(model_path),
        "input_size": int(metadata.get("input_size", 0)),
        "rows_used": int(metadata.get("rows_used", 0)),
        "threshold": float(metadata.get("threshold", 0.5)),
        "accuracy": float(metadata.get("accuracy", 0.0)),
        "precision": float(metadata.get("precision", 0.0)),
        "recall": float(metadata.get("recall", 0.0)),
        "f1": float(metadata.get("f1", 0.0)),
    }


async def predict_risk(request: RiskRequest) -> RiskResponse:
    """Predict crash risk for a location with a deterministic scoring scaffold."""
    settings = get_settings()

    warnings: list[str] = []
    weather = get_fallback_weather_snapshot()
    try:
        weather = await get_weather_snapshot(request.latitude, request.longitude)
    except (httpx.HTTPError, KeyError, ValueError):
        warnings.append("Live weather source unavailable; using fallback environmental data.")

    query_time = request.query_time_iso or datetime.now(tz=UTC)
    weather_condition = request.weather_condition or weather.short_forecast
    components, component_warnings = _build_components(
        latitude=request.latitude,
        longitude=request.longitude,
        road_class=request.road_class,
        weather_condition=weather_condition,
        temperature_f=weather.temperature_f,
    )
    warnings.extend(component_warnings)

    heuristic_score = _score_from_components(components)
    model_score, model_threshold = _infer_model_probability(
        request=request,
        weather_condition=weather_condition,
        query_time=query_time,
    )

    risk_score = heuristic_score
    if model_score is not None:
        risk_score = model_score
        if model_threshold is not None and model_score >= model_threshold:
            warnings.append(
                f"Model threshold alert: probability {model_score:.2f} exceeds "
                f"threshold {model_threshold:.2f}."
            )
    else:
        warnings.append("Model artifact unavailable; using deterministic fallback scoring.")

    score_100 = int(round(risk_score * 100))
    tier = _tier_from_score(score_100)

    road = request.road_name or "Unknown road segment"
    segment = request.segment or "Segment details unavailable"

    summary = (
        f"Model {settings.model_version} estimates a {score_100}% crash risk profile for "
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
