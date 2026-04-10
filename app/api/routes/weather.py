"""Weather data via Open-Meteo (forecast / current conditions)."""

import asyncio
from collections import OrderedDict
import copy
import time
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Global precision for coordinate rounding before external API calls.
COORDINATE_DECIMAL_PRECISION = 2
# Cache settings: 10 minutes TTL and a bounded in-memory size.
CACHE_TTL_SECONDS = 600
MAX_CACHE_ENTRIES = 200

# Human-readable labels for selected WMO weather_code values (see Open-Meteo docs).
_WEATHER_CODE_LABELS: dict[int, str] = {
    0: "1 - CLEAR",
    **{code: "2 - CLOUDY" for code in (1, 2, 3)},
    **{code: "3 - RAIN" for code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95)},
    **{code: "4 - SLEET/HAIL" for code in (96, 99)},
    **{code: "5 - SNOW" for code in (71, 73, 75, 77, 85, 86)},
    **{code: "6 - FOG" for code in (45, 48)},
}

# Uses the Open-Meteo weather code plus temperature to estimate the road condition.
def _get_road_code_label(weather_code: int, temperature: float | None) -> str:
    """Map the weather code and temperature to a road-condition label."""
    dry_codes = (0, 1, 2, 3, 45, 48)
    wet_codes = (51, 53, 55, 56, 57, 61, 63, 66, 80, 81, 95, 96, 99)
    standing_water_codes = (65, 67, 82)
    snow_codes = (71, 73, 75, 77, 85, 86)

    if weather_code in dry_codes:
        return "1 - DRY"

    if weather_code in snow_codes:
        return "4 - SNOW" 

    if weather_code in standing_water_codes:
        return "6 - ICE" if temperature is not None and temperature <= 0 else "3 - STANDING WATER"

    if weather_code in wet_codes:
        return "6 - ICE" if temperature is not None and temperature <= 0 else "2 - WET"

    return "99 - UNKNOWN"

router = APIRouter(prefix="/weather", tags=["weather"])

_weather_cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
_cache_locks: dict[str, asyncio.Lock] = {}


def _round_coordinate(value: float) -> float:
    """Round coordinates using the globally configured precision."""
    return round(value, COORDINATE_DECIMAL_PRECISION)


def _build_cache_key(latitude: float, longitude: float) -> str:
    """Build stable cache key from rounded coordinates."""
    return f"{latitude:.{COORDINATE_DECIMAL_PRECISION}f}:{longitude:.{COORDINATE_DECIMAL_PRECISION}f}"


def _get_cached_payload(cache_key: str) -> dict[str, Any] | None:
    """Return cached payload if present and not expired."""
    entry = _weather_cache.get(cache_key)
    if entry is None:
        return None

    expires_at, payload = entry
    now = time.monotonic()
    if now >= expires_at:
        _weather_cache.pop(cache_key, None)
        return None

    _weather_cache.move_to_end(cache_key)
    return copy.deepcopy(payload)


def _set_cached_payload(cache_key: str, payload: dict[str, Any]) -> None:
    """Store payload in cache with TTL and enforce max entry cap."""
    expires_at = time.monotonic() + CACHE_TTL_SECONDS
    _weather_cache[cache_key] = (expires_at, copy.deepcopy(payload))
    _weather_cache.move_to_end(cache_key)
    while len(_weather_cache) > MAX_CACHE_ENTRIES:
        _weather_cache.popitem(last=False)


def _with_cached_flag(payload: dict[str, Any], cached: bool) -> dict[str, Any]:
    """Return payload with a top-level cached flag."""
    payload_with_flag = copy.deepcopy(payload)
    payload_with_flag["cached"] = cached
    return payload_with_flag


def _enrich_current_with_weather_label(payload: dict[str, Any]) -> dict[str, Any]:
    """Add derived weather and road-condition labels to `current`, when present."""
    current = payload.get("current")
    if not isinstance(current, dict):
        return payload
    code = current.get("weather_code")
    if code is None:
        return payload
    try:
        code_int = int(code)
    except (TypeError, ValueError):
        return payload

    temperature = current.get("temperature_2m")
    try:
        temperature_float = float(temperature) if temperature is not None else None
    except (TypeError, ValueError):
        temperature_float = None

    normalized_code = code_int if code_int in _WEATHER_CODE_LABELS else 99
    if normalized_code != code_int:
        current["weather_code"] = 99

    current["weather"] = _WEATHER_CODE_LABELS.get(normalized_code, "99 - UNKNOWN")
    current["road_condition"] = _get_road_code_label(normalized_code, temperature_float)
    return payload


@router.get("/forecast")
async def get_forecast_current(
    latitude: float = Query(..., ge=-90.0, le=90.0, description="Latitude in degrees"),
    longitude: float = Query(..., ge=-180.0, le=180.0, description="Longitude in degrees"),
) -> dict[str, Any]:
    """
    Fetch current conditions from Open-Meteo for the given coordinates.

    Returns temperature, precipitation, WMO weather_code, and a derived `weather` label.
    """
    rounded_latitude = _round_coordinate(latitude)
    rounded_longitude = _round_coordinate(longitude)
    cache_key = _build_cache_key(rounded_latitude, rounded_longitude)

    cached_payload = _get_cached_payload(cache_key)
    if cached_payload is not None:
        return _with_cached_flag(cached_payload, cached=True)

    lock = _cache_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        cached_payload = _get_cached_payload(cache_key)
        if cached_payload is not None:
            return _with_cached_flag(cached_payload, cached=True)

        params = {
            "latitude": rounded_latitude,
            "longitude": rounded_longitude,
            "current": "temperature_2m,precipitation,weather_code",
            "timezone": "auto",
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(OPEN_METEO_FORECAST_URL, params=params)
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to reach weather service: {e!s}",
            ) from e

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Weather service returned {response.status_code}",
            )

        payload: dict[str, Any] = response.json()
        enriched_payload = _enrich_current_with_weather_label(payload)
        _set_cached_payload(cache_key, enriched_payload)
        return _with_cached_flag(enriched_payload, cached=False)
