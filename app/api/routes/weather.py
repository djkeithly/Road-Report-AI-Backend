"""Weather data via Open-Meteo (forecast / current conditions)."""

import asyncio
from collections import OrderedDict
import copy
from dataclasses import dataclass
import time
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query

# External API endpoint for fetching weather forecasts.
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Global precision for coordinate rounding before external API calls.
COORDINATE_DECIMAL_PRECISION = 2
# Cache settings: 10 minutes TTL and a bounded in-memory size.
CACHE_TTL_SECONDS = 600
MAX_CACHE_ENTRIES = 200
# Batch settings for unique cache misses.
BATCH_MAX_SIZE = 5
BATCH_MAX_WAIT_SECONDS = 2.0
BATCH_WAIT_TIMEOUT_SECONDS = 35.0

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

_weather_cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
_batch_pending: OrderedDict[str, "_PendingBatchItem"] = OrderedDict()
_batch_lock = asyncio.Lock()
_batch_flush_task: asyncio.Task[None] | None = None


@dataclass
class _PendingBatchItem:
    """A unique rounded coordinate waiting for the next batch call."""
    cache_key: str
    latitude: float
    longitude: float
    waiters: list[asyncio.Future[dict[str, Any]]]


def _round_coordinate(value: float) -> float:
    """Round coordinates using the globally configured precision."""
    return round(value, COORDINATE_DECIMAL_PRECISION)


def _build_cache_key(latitude: float, longitude: float) -> str:
    """Build stable cache key from rounded coordinates."""
    return f"{latitude:.{COORDINATE_DECIMAL_PRECISION}f}:{longitude:.{COORDINATE_DECIMAL_PRECISION}f}"


def _get_cached_payload(cache_key: str) -> Any | None:
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


def _set_cached_payload(cache_key: str, payload: Any) -> None:
    """Store payload in cache with TTL and enforce max entry cap."""
    expires_at = time.monotonic() + CACHE_TTL_SECONDS
    _weather_cache[cache_key] = (expires_at, copy.deepcopy(payload))
    _weather_cache.move_to_end(cache_key)
    while len(_weather_cache) > MAX_CACHE_ENTRIES:
        _weather_cache.popitem(last=False)


def _with_cached_flag(payload: Any, cached: bool) -> dict[str, Any]:
    """Return payload with a top-level cached flag."""
    payload_copy = copy.deepcopy(payload)
    if isinstance(payload_copy, dict):
        payload_copy["cached"] = cached
        return payload_copy
    return {"cached": cached, "data": payload_copy}


def _pop_next_batch_locked() -> list[_PendingBatchItem]:
    """Pop the next batch from the pending queue (requires _batch_lock)."""
    batch: list[_PendingBatchItem] = []
    while _batch_pending and len(batch) < BATCH_MAX_SIZE:
        _, pending_item = _batch_pending.popitem(last=False)
        batch.append(pending_item)
    return batch


async def _flush_after_delay() -> None:
    """Sleep for the batch window, then process queued requests."""
    global _batch_flush_task
    try:
        await asyncio.sleep(BATCH_MAX_WAIT_SECONDS)
    except asyncio.CancelledError:
        return

    batch_to_process: list[_PendingBatchItem] = []
    async with _batch_lock:
        _batch_flush_task = None
        batch_to_process = _pop_next_batch_locked()
        if _batch_pending and _batch_flush_task is None:
            _batch_flush_task = asyncio.create_task(_flush_after_delay())

    if batch_to_process:
        asyncio.create_task(_process_batch(batch_to_process))


def _extract_bulk_results(payload: Any, expected_count: int) -> list[Any]:
    """Normalize Open-Meteo responses into one result per requested coordinate."""
    if isinstance(payload, list):
        if len(payload) != expected_count:
            raise ValueError(
                f"Bulk response item count mismatch: expected {expected_count}, got {len(payload)}"
            )
        return payload

    if expected_count == 1:
        return [payload]

    raise ValueError(
        "Unexpected Open-Meteo bulk response format for multiple coordinates."
    )


async def _process_batch(batch: list[_PendingBatchItem]) -> None:
    """Execute one upstream bulk request and resolve all waiting callers."""
    global _batch_flush_task
    latitude_query = ",".join(
        f"{item.latitude:.{COORDINATE_DECIMAL_PRECISION}f}" for item in batch
    )
    longitude_query = ",".join(
        f"{item.longitude:.{COORDINATE_DECIMAL_PRECISION}f}" for item in batch
    )
    params = {
        "latitude": latitude_query,
        "longitude": longitude_query,
        "current": "temperature_2m,precipitation,weather_code",
        "timezone": "auto",
    }

    try:
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

        payload: Any = response.json()
        raw_items = _extract_bulk_results(payload, len(batch))
        for pending_item, raw_item in zip(batch, raw_items, strict=True):
            enriched_item = _enrich_payload(raw_item)
            _set_cached_payload(pending_item.cache_key, enriched_item)
            for waiter in pending_item.waiters:
                if not waiter.done():
                    waiter.set_result(_with_cached_flag(enriched_item, cached=False))
    except Exception as e:
        if isinstance(e, HTTPException):
            error = e
        else:
            error = HTTPException(status_code=502, detail=f"Bulk weather fetch failed: {e!s}")

        for pending_item in batch:
            for waiter in pending_item.waiters:
                if not waiter.done():
                    waiter.set_exception(error)
    finally:
        async with _batch_lock:
            if _batch_pending and _batch_flush_task is None:
                _batch_flush_task = asyncio.create_task(_flush_after_delay())


async def _enqueue_and_wait(cache_key: str, latitude: float, longitude: float) -> dict[str, Any]:
    """Queue one cache miss and wait for its batched upstream result."""
    global _batch_flush_task
    loop = asyncio.get_running_loop()
    waiter: asyncio.Future[dict[str, Any]] = loop.create_future()
    batch_to_process: list[_PendingBatchItem] | None = None

    async with _batch_lock:
        cached_payload = _get_cached_payload(cache_key)
        if cached_payload is not None:
            return _with_cached_flag(cached_payload, cached=True)

        pending_item = _batch_pending.get(cache_key)
        if pending_item is None:
            _batch_pending[cache_key] = _PendingBatchItem(
                cache_key=cache_key,
                latitude=latitude,
                longitude=longitude,
                waiters=[waiter],
            )
        else:
            pending_item.waiters.append(waiter)

        if len(_batch_pending) >= BATCH_MAX_SIZE:
            batch_to_process = _pop_next_batch_locked()
            if _batch_flush_task is not None:
                _batch_flush_task.cancel()
                _batch_flush_task = None
            if _batch_pending and _batch_flush_task is None:
                _batch_flush_task = asyncio.create_task(_flush_after_delay())
        elif _batch_flush_task is None:
            _batch_flush_task = asyncio.create_task(_flush_after_delay())

    if batch_to_process:
        asyncio.create_task(_process_batch(batch_to_process))

    try:
        return await asyncio.wait_for(waiter, timeout=BATCH_WAIT_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as e:
        async with _batch_lock:
            pending_item = _batch_pending.get(cache_key)
            if pending_item and waiter in pending_item.waiters:
                pending_item.waiters.remove(waiter)
                if not pending_item.waiters:
                    _batch_pending.pop(cache_key, None)
        raise HTTPException(status_code=504, detail="Timed out waiting for batched weather result.") from e


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


def _enrich_payload(payload: Any) -> Any:
    """Enrich single or bulk weather payloads."""
    if isinstance(payload, dict):
        return _enrich_current_with_weather_label(payload)
    if isinstance(payload, list):
        enriched_list: list[Any] = []
        for item in payload:
            if isinstance(item, dict):
                enriched_list.append(_enrich_current_with_weather_label(item))
            else:
                enriched_list.append(item)
        return enriched_list
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
    return await _enqueue_and_wait(cache_key, rounded_latitude, rounded_longitude)
