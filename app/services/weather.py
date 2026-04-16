"""weather.gov client service for risk enrichment."""

import httpx

from app.config import get_settings
from app.schemas.risk import WeatherSnapshot


async def get_weather_snapshot(latitude: float, longitude: float) -> WeatherSnapshot:
    """Fetch a weather snapshot for a location using weather.gov."""
    settings = get_settings()
    headers = {
        "User-Agent": settings.weather_user_agent,
        "Accept": "application/geo+json",
    }
    timeout = settings.weather_timeout_seconds
    base_url = settings.weather_api_base_url.rstrip("/")

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        point_url = f"{base_url}/points/{latitude},{longitude}"
        point_response = await client.get(point_url)
        point_response.raise_for_status()
        forecast_url = point_response.json()["properties"]["forecastHourly"]

        forecast_response = await client.get(forecast_url)
        forecast_response.raise_for_status()
        forecast_periods = forecast_response.json()["properties"]["periods"]

    if not forecast_periods:
        return WeatherSnapshot(source="weather.gov")

    first_period = forecast_periods[0]
    return WeatherSnapshot(
        shortForecast=first_period.get("shortForecast"),
        temperatureF=first_period.get("temperature"),
        windSpeed=first_period.get("windSpeed"),
        source="weather.gov",
    )


def get_fallback_weather_snapshot() -> WeatherSnapshot:
    """Return a fallback weather payload when live fetch fails."""
    return WeatherSnapshot(
        shortForecast=None,
        temperatureF=None,
        windSpeed=None,
        source="fallback",
    )
