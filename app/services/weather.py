"""weather providers for risk enrichment."""

import httpx

from app.config import get_settings
from app.schemas.risk import WeatherSnapshot

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_OPEN_METEO_LABELS: dict[int, str] = {
    0: "1 - CLEAR",
    **{code: "2 - CLOUDY" for code in (1, 2, 3)},
    **{code: "3 - RAIN" for code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95)},
    **{code: "4 - SLEET/HAIL" for code in (96, 99)},
    **{code: "5 - SNOW" for code in (71, 73, 75, 77, 85, 86)},
    **{code: "6 - FOG" for code in (45, 48)},
}


def _open_meteo_label(weather_code: int | None) -> str | None:
    """Map Open-Meteo weather codes to the same labels used in scoring."""
    if weather_code is None:
        return None
    return _OPEN_METEO_LABELS.get(int(weather_code), "99 - UNKNOWN")


async def _get_open_meteo_snapshot(latitude: float, longitude: float) -> WeatherSnapshot:
    """Fetch current weather from Open-Meteo as a fallback source."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,weather_code,wind_speed_10m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "auto",
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(OPEN_METEO_FORECAST_URL, params=params)
        response.raise_for_status()
        current = response.json().get("current", {})

    temperature = current.get("temperature_2m")
    wind_speed = current.get("wind_speed_10m")
    weather_code = current.get("weather_code")

    return WeatherSnapshot(
        shortForecast=_open_meteo_label(weather_code),
        temperatureF=(int(round(temperature)) if isinstance(temperature, (int, float)) else None),
        windSpeed=(f"{int(round(wind_speed))} mph" if isinstance(wind_speed, (int, float)) else None),
        source="open-meteo",
    )


async def get_weather_snapshot(latitude: float, longitude: float) -> WeatherSnapshot:
    """Fetch a weather snapshot for a location using weather.gov, then Open-Meteo fallback."""
    settings = get_settings()
    headers = {
        "User-Agent": settings.weather_user_agent,
        "Accept": "application/geo+json",
    }
    timeout = settings.weather_timeout_seconds
    base_url = settings.weather_api_base_url.rstrip("/")

    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            point_url = f"{base_url}/points/{latitude},{longitude}"
            point_response = await client.get(point_url)
            point_response.raise_for_status()
            forecast_url = point_response.json()["properties"]["forecastHourly"]

            forecast_response = await client.get(forecast_url)
            forecast_response.raise_for_status()
            forecast_periods = forecast_response.json()["properties"]["periods"]

        if forecast_periods:
            first_period = forecast_periods[0]
            return WeatherSnapshot(
                shortForecast=first_period.get("shortForecast"),
                temperatureF=first_period.get("temperature"),
                windSpeed=first_period.get("windSpeed"),
                source="weather.gov",
            )
    except (httpx.HTTPError, KeyError, ValueError, TypeError):
        pass

    return await _get_open_meteo_snapshot(latitude, longitude)


def get_fallback_weather_snapshot() -> WeatherSnapshot:
    """Return a fallback weather payload when live fetch fails."""
    return WeatherSnapshot(
        shortForecast=None,
        temperatureF=None,
        windSpeed=None,
        source="fallback",
    )
