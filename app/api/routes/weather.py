"""Weather data via Open-Meteo (forecast / current conditions)."""

from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

router = APIRouter(prefix="/weather", tags=["weather"])


@router.get("/forecast")
async def get_forecast_current(
    latitude: float = Query(..., ge=-90.0, le=90.0, description="Latitude in degrees"),
    longitude: float = Query(..., ge=-180.0, le=180.0, description="Longitude in degrees"),
) -> dict[str, Any]:
    """
    Fetch current conditions from Open-Meteo for the given coordinates.

    Returns temperature, precipitation, and WMO weather code for the current hour.
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

    return response.json()
