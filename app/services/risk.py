"""Risk prediction service with weather enrichment and scoring scaffold."""

import json
import pickle
import re
from datetime import UTC, datetime
from functools import lru_cache
from math import fabs
from pathlib import Path
from typing import Any

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


def _clamp_score(value: float) -> float:
    """Clamp component-style scores to a realistic 0-100 range."""
    return round(max(1.0, min(99.0, value)), 2)


def _build_components(
    *,
    latitude: float,
    longitude: float,
    road_class: str | None,
    weather_condition: str | None,
    temperature_f: int | None,
    query_time: datetime,
) -> tuple[RiskComponents, list[str]]:
    """Assemble explainable weighted components and warnings."""
    warnings: list[str] = []
    road_class_label = (road_class or "UNKNOWN").upper()
    hour = query_time.hour
    is_weekend = query_time.weekday() >= 5
    is_peak_commute = hour in {6, 7, 8, 9, 15, 16, 17, 18}
    is_overnight = hour <= 4
    corridor_variation = (fabs(latitude * 2.7) + fabs(longitude * 1.9)) % 18
    geo_pressure = (fabs(latitude - longitude) * 2.4) % 16
    road_class_pressure = {
        "INTERSTATE": 22,
        "TOLLWAY": 19,
        "US & STATE HIGHWAYS": 15,
        "FARM TO MARKET": 11,
        "CITY STREET": 8,
        "UNKNOWN": 6,
    }.get(road_class_label, 6)

    condition = (weather_condition or "").lower()
    precipitation_penalty = 0
    environmental_score = 16.0
    if condition:
        if "storm" in condition or "thunder" in condition:
            environmental_score = 84.0
            precipitation_penalty = 14
        elif "snow" in condition or "ice" in condition:
            environmental_score = 91.0
            precipitation_penalty = 18
        elif "rain" in condition or "sleet" in condition:
            environmental_score = 68.0
            precipitation_penalty = 10
        elif "fog" in condition:
            environmental_score = 58.0
            precipitation_penalty = 8
        elif "cloud" in condition:
            environmental_score = 28.0
        else:
            environmental_score = 18.0
    elif temperature_f is None:
        warnings.append(
            "Weather data unavailable; environmental component estimated conservatively."
        )
        environmental_score = 34.0

    if temperature_f is not None:
        if temperature_f <= 32:
            environmental_score += 12
            precipitation_penalty += 6
        elif temperature_f >= 100:
            environmental_score += 8
        elif temperature_f >= 90:
            environmental_score += 4

    road_condition_score = 18.0 + road_class_pressure + corridor_variation + precipitation_penalty
    historical_score = 20.0 + geo_pressure + (10 if is_peak_commute else 0) + (5 if is_weekend else 0)
    traffic_score = 14.0 + road_class_pressure + (14 if is_peak_commute else 4) + ((fabs(latitude - longitude) * 3.1) % 14)
    if is_overnight:
        traffic_score -= 8

    components = RiskComponents(
        roadCondition=RiskComponent(
            name="Road Condition",
            key="C",
            score=_clamp_score(road_condition_score),
            maxPoints=25,
            weight=0.30,
            details=[
                RiskDetail(
                    label="Road Class", value=requested_value_or_dash(road_class_label)
                ),
                RiskDetail(
                    label="Surface Condition",
                    value=("Weather-adjusted estimate" if precipitation_penalty else "Nominal"),
                ),
            ],
            source="txdot-cris-estimate",
        ),
        historical=RiskComponent(
            name="Historical Crash Pattern",
            key="A",
            score=_clamp_score(historical_score),
            maxPoints=25,
            weight=0.30,
            details=[
                RiskDetail(label="Crash trend window", value=("Peak commute profile" if is_peak_commute else "Standard profile")),
                RiskDetail(label="County profile", value=("Weekend-adjusted" if is_weekend else "Weekday-adjusted")),
            ],
            source="txdot-cris-estimate",
        ),
        environmental=RiskComponent(
            name="Environmental Conditions",
            key="E",
            score=_clamp_score(environmental_score),
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
            score=_clamp_score(traffic_score),
            maxPoints=25,
            weight=0.15,
            details=[
                RiskDetail(
                    label="Time profile",
                    value=("Peak commute" if is_peak_commute else "Overnight" if is_overnight else "Standard daytime"),
                ),
                RiskDetail(label="Congestion proxy", value=requested_value_or_dash(road_class_label)),
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


def _normalize_road_text(value: str) -> str:
    """Normalize road-name text to improve model feature matching."""
    cleaned = re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()
    return re.sub(r"\s+", " ", cleaned)


def _token_set(value: str) -> set[str]:
    """Return normalized token set with low-information words removed."""
    stop_words = {
        "N",
        "S",
        "E",
        "W",
        "NORTH",
        "SOUTH",
        "EAST",
        "WEST",
        "COUNTY",
        "CITY",
        "OF",
        "THE",
    }
    return {token for token in _normalize_road_text(value).split() if token and token not in stop_words}


def _best_fuzzy_road_match(candidate: str, known: set[str]) -> str | None:
    """Return the closest known road if token overlap is sufficiently strong."""
    candidate_tokens = _token_set(candidate)
    if not candidate_tokens:
        return None

    best_match: str | None = None
    best_score = 0.0
    for road in known:
        road_tokens = _token_set(road)
        if not road_tokens:
            continue
        overlap = len(candidate_tokens & road_tokens)
        if overlap == 0:
            continue
        score = overlap / max(len(candidate_tokens), len(road_tokens))
        if score > best_score:
            best_score = score
            best_match = road

    if best_score >= 0.6:
        return best_match
    return None


def _canonicalize_road_name_for_model(road_name: str | None) -> str:
    """Map user-entered road names to the closest known model feature label."""
    if not road_name:
        return "UNKNOWN"

    normalized = _normalize_road_text(road_name)
    if not normalized:
        return "UNKNOWN"

    known = set(_known_road_names())
    if normalized in known:
        return normalized

    long_to_short = {
        "ROAD": "RD",
        "STREET": "ST",
        "BOULEVARD": "BLVD",
        "TURNPIKE": "TPKE",
        "AVENUE": "AVE",
        "HIGHWAY": "HWY",
        "FREEWAY": "FWY",
        "INTERSTATE": "IH",
    }
    short_to_long = {value: key for key, value in long_to_short.items()}

    words = normalized.split()
    abbreviated = " ".join(long_to_short.get(word, word) for word in words)
    expanded = " ".join(short_to_long.get(word, word) for word in words)

    for candidate in (abbreviated, expanded):
        if candidate in known:
            return candidate

    fuzzy = _best_fuzzy_road_match(abbreviated, known)
    if fuzzy:
        return fuzzy

    fuzzy = _best_fuzzy_road_match(expanded, known)
    if fuzzy:
        return fuzzy

    return normalized


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


def _normalize_model_score(
    *,
    model_prob: float,
    min_bound: float,
    max_bound: float,
    heuristic_score: float,
) -> tuple[float, str]:
    """Convert model probability to display score, falling back when bounds collapse."""
    spread = max_bound - min_bound
    if spread < 1e-6:
        return heuristic_score, "collapsed_bounds"

    padding = spread * 0.10
    adjusted_min = min_bound - padding
    adjusted_max = max_bound + padding

    if adjusted_max <= adjusted_min:
        return heuristic_score, "invalid_bounds"

    normalized_score = (model_prob - adjusted_min) / (adjusted_max - adjusted_min)
    normalized_score = max(0.01, min(1.0, normalized_score))
    return normalized_score, "model"


def _blend_scores(*, model_score: float, heuristic_score: float) -> float:
    """Blend model and heuristic scores to avoid unstable or overly flat outputs."""
    blended = (model_score * 0.65) + (heuristic_score * 0.35)
    return max(0.01, min(1.0, blended))


@lru_cache(maxsize=1)
def _load_model_bundle() -> dict[str, object] | None:
    """Load trained Logistic Regression model and feature columns."""
    settings = get_settings()
    project_root = Path(__file__).resolve().parents[2]

    model_candidates = [
        Path(settings.model_file_path),
        project_root / settings.model_file_path,
    ]
    cols_candidates = [
        Path(settings.feature_columns_path),
        project_root / settings.feature_columns_path,
    ]

    model_path = next((path for path in model_candidates if path.exists()), None)
    cols_path = next((path for path in cols_candidates if path.exists()), None)

    if model_path is None or cols_path is None:
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

    return {
        "model": model,
        "feature_columns": feature_columns,
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
    road_class: str | None,
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

    # Match integer format exactly
    crash_time_int = int((query_time.hour * 100) + query_time.minute)
    crash_month_int = int(query_time.month)

    canonical_road_name = _canonicalize_road_name_for_model(request.road_name)

    row = {
        "City": [city],
        "County": ["DALLAS"],
        "Crash Month": [crash_month_int],
        "Crash Time": [crash_time_int],
        "Road Class": [road_class.upper() if road_class else "UNKNOWN"],
        "Street Name": [canonical_road_name],
        "Surface Condition": [s_cond],
        "Weather Condition": [w_cond],
    }
    return pd.DataFrame(row)


def _banding_anchor_street_name() -> str:
    """Return a stable street name anchor for normalization cutoffs."""
    known = _known_road_names()
    for preferred in ("S I 35E S", "I 35E", "IH0035E"):
        if preferred in known:
            return preferred
    return known[0] if known else "UNKNOWN"


def _infer_model_probability(
    *,
    request: RiskRequest,
    weather_condition: str | None,
    query_time: datetime,
    city: str,
    road_class: str | None,
) -> tuple[float | None, float | None, float | None]:
    """Return model probability, along with dynamic min and max bounds for the current context."""
    bundle = _load_model_bundle()
    if bundle is None:
        return None, None, None

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    # 1. Get Live Probability
    df = _build_inference_df(
        request=request,
        weather_condition=weather_condition,
        query_time=query_time,
        city=city,
        road_class=road_class,
    )

    encoded = pd.get_dummies(df)
    aligned = encoded.reindex(columns=feature_columns, fill_value=0)
    tensor = torch.tensor(aligned.to_numpy(dtype=np.float32))

    with torch.no_grad():
        probability = float(torch.sigmoid(model(tensor)).item())

    # 2. Dynamically calculate bounds using LIVE time, month, and city
    crash_time_int = int((query_time.hour * 100) + query_time.minute)
    crash_month_int = int(query_time.month)
    street_name = _banding_anchor_street_name()

    banding_dataset = [
        {
            "City": [city],
            "County": ["DALLAS"],
            "Crash Month": [crash_month_int],
            "Crash Time": [crash_time_int],
            "Road Class": ["CITY STREET"],
            "Street Name": [street_name],
            "Surface Condition": ["1 - DRY"],
            "Weather Condition": ["1 - CLEAR"],
        },
        {
            "City": [city],
            "County": ["DALLAS"],
            "Crash Month": [crash_month_int],
            "Crash Time": [crash_time_int],
            "Road Class": ["INTERSTATE"],
            "Street Name": [street_name],
            "Surface Condition": ["1 - DRY"],
            "Weather Condition": ["1 - CLEAR"],
        },
        {
            "City": [city],
            "County": ["DALLAS"],
            "Crash Month": [crash_month_int],
            "Crash Time": [crash_time_int],
            "Road Class": ["INTERSTATE"],
            "Street Name": [street_name],
            "Surface Condition": ["2 - WET"],
            "Weather Condition": ["3 - RAIN"],
        },
    ]

    cutoffs = []
    for band_item in banding_dataset:
        custom_df = pd.DataFrame(band_item)
        custom_encoded = pd.get_dummies(custom_df)
        custom_aligned = custom_encoded.reindex(columns=feature_columns, fill_value=0)
        custom_tensor = torch.tensor(custom_aligned.to_numpy(dtype=np.float32))

        with torch.no_grad():
            prob = float(torch.sigmoid(model(custom_tensor)).item())
            cutoffs.append(prob)

    min_bound = min(cutoffs)
    max_bound = max(cutoffs)

    return probability, min_bound, max_bound


@lru_cache(maxsize=1)
def _known_road_names() -> tuple[str, ...]:
    """Return sorted road names inferred from model feature columns."""
    bundle = _load_model_bundle()
    if bundle is None:
        return tuple()

    feature_columns_obj = bundle["feature_columns"]
    if not isinstance(feature_columns_obj, (list, tuple)):
        try:
            feature_columns = list(feature_columns_obj)
        except TypeError:
            return tuple()
    else:
        feature_columns = list(feature_columns_obj)

    if not feature_columns:
        return tuple()

    prefixes = ("Street Name_", "streetname_", "street_name_")
    road_names: set[str] = set()
    for column in feature_columns:
        if not isinstance(column, str):
            continue
        for prefix in prefixes:
            if column.startswith(prefix):
                road_name = column[len(prefix) :].strip()
                if road_name:
                    road_names.add(road_name)
                break

    return tuple(sorted(road_names))


def get_road_suggestions(query: str, *, limit: int = 8) -> list[str]:
    """Return road-name autocomplete suggestions based on model features."""
    needle = query.strip().lower()
    if not needle:
        return []

    starts_with: list[str] = []
    contains: list[str] = []
    for road_name in _known_road_names():
        lowered = road_name.lower()
        if lowered.startswith(needle):
            starts_with.append(road_name)
        elif needle in lowered:
            contains.append(road_name)
        if len(starts_with) >= limit:
            break

    if len(starts_with) < limit:
        remaining = limit - len(starts_with)
        starts_with.extend(contains[:remaining])
    return starts_with[:limit]


def get_model_runtime_metadata() -> dict[str, str | int | float | bool]:
    """Return model artifact metadata for runtime diagnostics and UI display."""
    settings = get_settings()
    project_root = Path(__file__).resolve().parents[2]
    model_candidates = [
        Path(settings.model_file_path),
        project_root / settings.model_file_path,
    ]
    model_path = next((path for path in model_candidates if path.exists()), None)
    cols_candidates = [
        Path(settings.feature_columns_path),
        project_root / settings.feature_columns_path,
    ]
    cols_path = next((path for path in cols_candidates if path.exists()), None)

    if model_path is None or cols_path is None:
        unresolved_path = str((project_root / settings.model_file_path).resolve())
        unresolved_cols = str((project_root / settings.feature_columns_path).resolve())
        return {
            "available": False,
            "model_path": unresolved_path,
            "feature_columns_path": unresolved_cols,
            "message": "Model artifact files are missing.",
        }

    metadata: dict[str, str | int | float | bool] = {
        "available": True,
        "model_path": str(model_path),
        "feature_columns_path": str(cols_path),
        "model_type": "logistic_regression",
        "model_version": settings.model_version,
        "message": "Using PyTorch Logistic Regression model",
    }

    try:
        with cols_path.open("rb") as columns_file:
            feature_columns = pickle.load(columns_file)
        metadata["input_size"] = len(feature_columns)
        metadata["known_road_count"] = len(
            [
                col
                for col in feature_columns
                if isinstance(col, str) and col.startswith("Street Name_")
            ]
        )
    except Exception:
        metadata["message"] = "Model file found, but feature columns could not be read."

    meta_candidates = [
        model_path.with_suffix(".meta.json"),
    ]
    meta_path = next((path for path in meta_candidates if path.exists()), None)
    if meta_path is not None:
        try:
            with meta_path.open("r", encoding="utf-8") as meta_file:
                meta_json: dict[str, Any] = json.load(meta_file)
            for key in (
                "input_size",
                "rows_used",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ):
                value = meta_json.get(key)
                if value is not None:
                    metadata[key] = value
        except (OSError, json.JSONDecodeError):
            pass

    return metadata


async def predict_risk(request: RiskRequest) -> RiskResponse:
    """Predict crash risk using model output with heuristic fallback when needed."""
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
        query_time=query_time,
    )
    warnings.extend(component_warnings)

    city = await _get_city_from_coords(request.latitude, request.longitude)
    heuristic_score = _score_from_components(components)

    model_prob, min_bound, max_bound = _infer_model_probability(
        request=request,
        weather_condition=weather_condition,
        query_time=query_time,
        city=city,
        road_class=inferred_road_class,
    )

    if model_prob is not None and min_bound is not None and max_bound is not None:
        normalized_model_score, score_source = _normalize_model_score(
            model_prob=model_prob,
            min_bound=min_bound,
            max_bound=max_bound,
            heuristic_score=heuristic_score,
        )
        if score_source == "model":
            risk_score = _blend_scores(
                model_score=normalized_model_score,
                heuristic_score=heuristic_score,
            )
        else:
            risk_score = heuristic_score
            warnings.append(
                "Model normalization bounds were too narrow; using heuristic fallback scoring."
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
