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


class StationPlace(TypedDict):
    name: str
    lat: float | None
    lon: float | None


def _csv_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_csv_path(relative_or_absolute: str | Path) -> Path:
    p = Path(relative_or_absolute)
    return p if p.is_absolute() else _csv_root() / p


def _lowercase_column_map(df: pd.DataFrame) -> dict[str, str]:
    return {str(c).strip().lower(): c for c in df.columns}


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


def _nearest_station_name(
    lat: float,
    lon: float,
    stations: list[StationPlace],
) -> str:
    """Pick the ``stations`` row with coordinates nearest to ``(lat, lon)``; return its ``name``."""
    best_name = ""
    best_d2: float | None = None
    for s in stations:
        slat, slon = s.get("lat"), s.get("lon")
        if slat is None or slon is None:
            continue
        d2 = (float(slat) - lat) ** 2 + (float(slon) - lon) ** 2
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_name = s["name"]
    return best_name


def first_station_name(stations: list[StationPlace]) -> str:
    """Return the ``name`` of the first row in ``stations`` (empty if there are none)."""
    return stations[0]["name"] if stations else ""


def append_compared_places_to_stations_csv(
    places: list[str],
    compared_coords: list[tuple[float, float] | None],
    stations: list[StationPlace],
    *,
    stations_csv: str | Path = DEFAULT_STATIONS_CSV,
) -> int:
    """
    For each ``places[i]``, append one line to ``stations.csv``:

    - If ``compared_coords[i]`` is a ``(lat, lon)`` tuple: ``lat``/``lon`` from it,
      ``closest_station`` = nearest row in ``stations`` by coordinates.
    - If geocoding failed (``compared_coords[i]`` is ``None``): ``lat`` and ``lon``
      are left blank, ``closest_station`` = ``first_station_name(stations)``.

    ``station_flags`` is always ``0`` for appended rows.

    Returns the number of rows appended.
    """
    if len(places) != len(compared_coords):
        raise ValueError("places and compared_coords must have the same length.")

    rows: list[dict[str, object]] = []
    default_closest = first_station_name(stations)
    for place, coord in zip(places, compared_coords):
        if coord is None:
            rows.append(
                {
                    "name": place,
                    "lat": "",
                    "lon": "",
                    "closest_station": default_closest,
                    "station_flags": 0,
                }
            )
            continue

        lat, lon = coord
        closest = _nearest_station_name(lat, lon, stations)
        rows.append(
            {
                "name": place,
                "lat": lat,
                "lon": lon,
                "closest_station": closest,
                "station_flags": 0,
            }
        )

    if not rows:
        return 0

    path = _resolve_csv_path(stations_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows, columns=["name", "lat", "lon", "closest_station", "station_flags"])
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(path, index=False)
    else:
        new_df.to_csv(path, index=False)
    return len(rows)

class ComparedCoord(TypedDict):
    lat: float
    lon: float
    name: str


CITIES_CSV = "weather/weather_csvs/cities.csv"


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


def lookup(
    src_path: str | Path,
    cities: bool,
) -> bool:
    """
    Print unique place names from ``src_path``, geocode each (Nominatim),
    print station rows from ``stations_csv``, then append matched rows to
    ``stations.csv`` (see ``append_compared_places_to_stations_csv``).
    """

    # Read unique place names from the specified column in the source CSV.
    places = read_unique_places_from_csv(src_path, cities=cities)

    # Get all the existing stations from stations.csv
    # stations = read_stations_places(stations_csv)

    # if len(stations) == 0:
    #     print("No existing stations, catastrophic error, canceling data generation")
    #     return False

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
    
    # print("Necessary lookups complete, making station comparisons and appending to stations.csv...")

    # # Drop places that already appear as a station ``name`` (do not conflate with geocode failure).
    # existing_names = {s["name"].strip().casefold() for s in stations}
    # filtered_places: list[str] = []
    # filtered_coords: list[tuple[float, float] | None] = []
    # for place, coord in zip(places, compared_coords):
    #     if place.strip().casefold() in existing_names:
    #         print(f"'{place}' already has an entry in {stations_csv}, skipping.")
    #         continue
    #     filtered_places.append(place)
    #     filtered_coords.append(coord)

    # Append new rows to stations.csv for places that don't already have entries.
    # n = append_compared_places_to_stations_csv(
    #     filtered_places, filtered_coords, stations, stations_csv=stations_csv
    # )

    write_cities_csv(places, compared_coords)

    print("Geocoding pass complete (stations.csv append is commented out).")
    return True

