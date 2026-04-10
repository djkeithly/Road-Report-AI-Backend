"""Weather data via Open-Meteo (forecast / current conditions)."""

from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

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
    params = {
        "latitude": latitude,
        "longitude": longitude,
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
    return _enrich_current_with_weather_label(payload)
