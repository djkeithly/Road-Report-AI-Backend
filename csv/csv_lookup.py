"""
Read place names from a ``city`` / ``county`` column, geocode them, match to
``stations.csv`` coordinates, and append rows to ``stations.csv``.

Paths are resolved relative to this file's directory (``csv/``) when not
absolute — e.g. ``weather/weather_csvs/stations.csv``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TypedDict

import pandas as pd
import requests

_CITY = "city"
_COUNTY = "county"
DEFAULT_STATIONS_CSV = "weather/weather_csvs/stations.csv"

#   #   #   #   #   #   #   #   #
#       Helper functions        #
#   #   #   #   #   #   #   #   #
class StationPlace(TypedDict):
    name: str
    lat: float | None
    lon: float | None

class ComparedCoord(TypedDict):
    lat: float
    lon: float
    name: str

CITIES_CSV = "weather/weather_csvs/cities.csv"

def _csv_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_csv_path(relative_or_absolute: str | Path) -> Path:
    p = Path(relative_or_absolute)
    return p if p.is_absolute() else _csv_root() / p


def _lowercase_column_map(df: pd.DataFrame) -> dict[str, str]:
    return {str(c).strip().lower(): c for c in df.columns}

#   #   #   #   #   #   #   #   #   #   #
#       CSV reading and geocoding       #
#   #   #   #   #   #   #   #   #   #   #

def read_unique_places_from_csv(src_path: str | Path, *, cities: bool) -> list[str]:
    """
    Load ``src_path``, take values from the ``city`` or ``county`` column
    (depending on ``cities``), and return a list with no duplicates (order preserved).
    """
    path = _resolve_csv_path(src_path)
    df = pd.read_csv(path)
    want = _CITY if cities else _COUNTY
    lower = _lowercase_column_map(df)
    if want not in lower:
        raise ValueError(
            f"Expected a '{want}' column in {path}. Found columns: {list(df.columns)}"
        )
    col = lower[want]
    series = df[col].dropna().astype(str).str.strip()
    series = series[series != ""]
    return list(dict.fromkeys(series))


def geocode_city_state(city: str, state: str = "Texas") -> dict | None:
    """Geocode a city and state using Nominatim. Returns a dict with ``lat``, ``lon``, and ``display_name`` if successful, or ``None`` if geocoding failed."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "city": city,
        "state": state,
        "country": "USA",
        "format": "json",
        "limit": 1,
    }
    headers = {"User-Agent": "Road-Report-AI-Backend (DJKeithly3@gmail.com)"}
    response = requests.get(url, params=params, headers=headers, timeout=30)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None
    data = response.json()
    if not data:
        return None
    result = data[0]
    return {
        "lat": float(result["lat"]),
        "lon": float(result["lon"]),
        "display_name": result["display_name"],
    }

def write_cities_csv(
    places: list[str],
    compared_coords: list[ComparedCoord | tuple[float, float] | None],
) -> Path:
    """
    Write ``places`` and coordinates to ``weather/weather_csvs/cities.csv``.

    Each row uses the corresponding ``places`` string as ``name`` (not any
    ``name`` field on the coordinate object). ``compared_coords[i]`` may be a
    ``ComparedCoord`` dict, a ``(lat, lon)`` tuple, or ``None`` (blank lat/lon).

    ``places`` and ``compared_coords`` must have the same length.
    """
    if len(places) != len(compared_coords):
        raise ValueError("places and compared_coords must have the same length.")

    rows: list[dict[str, object]] = []
    for place, item in zip(places, compared_coords):
        if item is None:
            print(f"Could not geocode '{place}', skipping coordinates.")
            continue
        elif isinstance(item, tuple):
            lat, lon = item
            rows.append({"name": place, "lat": lat, "lon": lon})
        else:
            rows.append({"name": place, "lat": item["lat"], "lon": item["lon"]})

    path = _resolve_csv_path(CITIES_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["name", "lat", "lon"]).to_csv(path, index=False)
    return path

#   #   #   #   #   #   #   #   #   #   #   #
#       Main lookup and geocoding flow      # 
#   #   #   #   #   #   #   #   #`  #   #   #

def lookup(
    src_path: str | Path,
    cities: bool,
) -> bool:
    """
    Print unique place names from ``src_path``, geocode each (Nominatim),
    print station rows from ``stations_csv``, then append matched rows to
    ``cities.csv``.
    """

    # Read unique place names from the specified column in the source CSV.
    places = read_unique_places_from_csv(src_path, cities=cities)

    # Download coordinates from geocoding
    compared_coords: list[tuple[float, float] | None] = []
    for place in places:
        coords = geocode_city_state(place)
        if coords is not None:
            compared_coords.append((coords["lat"], coords["lon"]))
        else:
            print(f"Could not geocode '{place}'.")
            compared_coords.append(None)
        time.sleep(1)

    write_cities_csv(places, compared_coords)

    print("Geocoding pass complete.")
    return True

