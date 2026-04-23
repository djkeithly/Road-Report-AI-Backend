"""
Read place names from a ``city`` / ``county`` column and station rows from
``stations.csv``. Paths are resolved relative to this file's directory (``csv/``)
when not absolute — e.g. ``weather/weather_csvs/stations.csv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict
import requests

import pandas as pd

_CITY = "city"
_COUNTY = "county"
DEFAULT_STATIONS_CSV = "weather/weather_csvs/stations.csv"

# TypedDict for station records read from stations.csv. Only name, lat, lon are kept.
class StationPlace(TypedDict):
    name: str
    lat: float | None
    lon: float | None

# Helper functions for path resolution, column name normalization, and CSV reading.
def _csv_root() -> Path:
    return Path(__file__).resolve().parent

# Resolve a path that may be relative to the CSV directory or absolute.
def _resolve_csv_path(relative_or_absolute: str | Path) -> Path:
    p = Path(relative_or_absolute)
    return p if p.is_absolute() else _csv_root() / p

# Filter the ISD history dataframe by country and/or state, returning a copy of the filtered dataframe.
def _lowercase_column_map(df: pd.DataFrame) -> dict[str, str]:
    return {str(c).strip().lower(): c for c in df.columns}

# Read unique place names from the specified column in the CSV, returning a list of unique, non-empty strings.
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


import requests
import time

def geocode_city_state(city, state="Texas"):
    url = "https://nominatim.openstreetmap.org/search"
    
    params = {
        "city": city,
        "state": state,
        "country": "USA",
        "format": "json",
        "limit": 1
    }
    
    headers = {
        "User-Agent": "Road-Report-AI-Backend (DJKeithly3@gmail.com)"
    }

    response = requests.get(url, params=params, headers=headers)
    
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
        "display_name": result["display_name"]
    }



# Read station records from stations.csv, returning a list of dicts with name, lat, lon.
def read_stations_places(
    relative_path: str | Path = DEFAULT_STATIONS_CSV,
) -> list[StationPlace]:
    """
    Read ``stations.csv`` (default ``weather/weather_csvs/stations.csv``) and
    return ``{name, lat, lon}`` per row (other columns ignored).

    ``lat`` / ``lon`` are floats when parseable, otherwise ``None``.
    """
    path = _resolve_csv_path(relative_path)
    df = pd.read_csv(path)
    lower = _lowercase_column_map(df)
    for key in ("name", "lat", "lon"):
        if key not in lower:
            raise ValueError(
                f"Expected a '{key}' column in {path}. Found columns: {list(df.columns)}"
            )
    name_c, lat_c, lon_c = lower["name"], lower["lat"], lower["lon"]
    out: list[StationPlace] = []
    for _, row in df.iterrows():
        raw_name = row.get(name_c)
        name = str(raw_name).strip() if pd.notna(raw_name) else ""
        if not name:
            continue
        lat_n = pd.to_numeric(row.get(lat_c), errors="coerce")
        lon_n = pd.to_numeric(row.get(lon_c), errors="coerce")
        out.append(
            {
                "name": name,
                "lat": float(lat_n) if pd.notna(lat_n) else None,
                "lon": float(lon_n) if pd.notna(lon_n) else None,
            }
        )
    return out


def lookup(
    src_path: str | Path,
    cities: bool,
    *,
    stations_csv: str | Path = DEFAULT_STATIONS_CSV,
) -> None:
    """
    Main entry: print unique place names from ``src_path``, then print each
    station record from ``stations_csv``.
     """
    places = read_unique_places_from_csv(src_path, cities=cities)
    for place in places:
        print(place)

    for place in places:
        coords = geocode_city_state(place)
        print(coords)
        time.sleep(1)  # Get limited really fast if we don't sleep

    stations = read_stations_places(stations_csv)
    #for rec in stations:
        # print(rec)
    
    return None
