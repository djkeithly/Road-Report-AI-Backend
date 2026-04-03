"""
ISD station history: reverse-geocode LAT/LON with the U.S. Census coordinates
geocoder and write CSV with ``Geocoded_City``, ``Geocoded_County``, and
``Geocoded_County_Subdivision``.

Intended to be imported (e.g. from training pipelines), not run as a script.
Call ``geocode()`` with optional ``ctry`` / ``state`` filters on ``CTRY`` /
``STATE`` columns.

See: https://geocoding.geo.census.gov/geocoder/
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

CENSUS_COORD_GEOCODER = (
    "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
)


def _float_lat_lon(value) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    return float(s.replace("+", ""))


def census_reverse_geocode_city_county(
    longitude: float,
    latitude: float,
    *,
    timeout_sec: float = 30.0,
) -> tuple[str | None, str | None, str | None]:
    """
    Reverse-geocode a US point with the Census Bureau's public geocoder.
    Returns (city, county, county_subdivision). City is the incorporated place
    BASENAME when the point lies inside one; otherwise None.
    """
    params = urllib.parse.urlencode(
        {
            "x": longitude,
            "y": latitude,
            "benchmark": "Public_AR_Current",
            "vintage": "Current_Current",
            "format": "json",
        }
    )
    url = f"{CENSUS_COORD_GEOCODER}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None, None, None

    try:
        geographies = payload["result"]["geographies"]
    except (KeyError, TypeError):
        return None, None, None

    county_name = None
    counties = geographies.get("Counties") or []
    if counties:
        county_name = counties[0].get("NAME")

    city_name = None
    places = geographies.get("Incorporated Places") or []
    if places:
        city_name = places[0].get("BASENAME") or places[0].get("NAME")

    subdiv_name = None
    subdivs = geographies.get("County Subdivisions") or []
    if subdivs:
        subdiv_name = subdivs[0].get("NAME")

    return city_name, county_name, subdiv_name


def add_geocoded_city_county_columns(
    df: pd.DataFrame,
    *,
    lat_col: str = "LAT",
    lon_col: str = "LON",
    delay_sec: float = 0.12,
    timeout_sec: float = 30.0,
) -> pd.DataFrame:
    """
    Add Geocoded_City, Geocoded_County, and Geocoded_County_Subdivision using
    the Census geocoder. Rows with missing latitude or longitude are left blank.
    Duplicate coordinates reuse a small in-memory cache.
    """
    out = df.copy()
    cities: list[str | None] = []
    counties: list[str | None] = []
    subdivs: list[str | None] = []
    cache: dict[tuple[float, float], tuple[str | None, str | None, str | None]] = {}

    for _, row in out.iterrows():
        lat = _float_lat_lon(row.get(lat_col))
        lon = _float_lat_lon(row.get(lon_col))
        if lat is None or lon is None:
            cities.append(None)
            counties.append(None)
            subdivs.append(None)
            continue

        print(f"Geocoding LAT={lat} LON={lon}...")

        key = (round(lat, 6), round(lon, 6))
        if key not in cache:
            city, county, subdiv = census_reverse_geocode_city_county(
                lon, lat, timeout_sec=timeout_sec
            )
            cache[key] = (city, county, subdiv)
            if delay_sec > 0:
                time.sleep(delay_sec)
        else:
            city, county, subdiv = cache[key]

        cities.append(city)
        counties.append(county)
        subdivs.append(subdiv)

    out["Geocoded_City"] = cities
    out["Geocoded_County"] = counties
    out["Geocoded_County_Subdivision"] = subdivs
    return out


def filter_isd_by_country_state(
    df: pd.DataFrame,
    ctry: str | None,
    state: str | None,
) -> pd.DataFrame:
    """
    Apply ``CTRY`` / ``STATE`` filters. Both None means no filtering (all rows).

    - Both set: rows where CTRY and STATE match.
    - Only ctry: rows where CTRY matches.
    - Only state: rows where STATE matches.
    """
    if "CTRY" not in df.columns or "STATE" not in df.columns:
        raise ValueError("Input must have 'CTRY' and 'STATE' columns.")

    if ctry is None and state is None:
        return df.copy()

    if ctry is not None and state is not None:
        return df[(df["CTRY"] == ctry) & (df["STATE"] == state)].copy()

    if ctry is not None:
        return df[df["CTRY"] == ctry].copy()

    return df[df["STATE"] == state].copy()


def geocode(
    src: str = "isd-history.csv",
    dst: str = "geocoded.csv",
    ctry: str = "US",
    state: str = "TX",
    *,
    delay_sec: float = 0.12,
    timeout_sec: float = 30.0,
) -> str | None:
    """
    Read ISD history CSV, optionally filter by country/state, geocode rows,
    and write ``dst`` (paths relative to this file's directory unless absolute).

    Filtering (matches ``CTRY`` / ``STATE`` columns in the file):

    - ``ctry`` and ``state`` both set: only rows with both values.
    - Only ``ctry``: only rows with that country.
    - Only ``state``: only rows with that state.
    - Both ``None``: geocode every row in the file.

    The Census geocoder is intended for U.S. coordinates; rows outside the U.S.
    may get empty geocoded fields even when LAT/LON are present.

    Returns:
        Absolute path written, or None if no rows matched the filter.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = src if os.path.isabs(src) else os.path.join(script_dir, src)
    dst_path = dst if os.path.isabs(dst) else os.path.join(script_dir, dst)

    df = pd.read_csv(src_path)
    subset = filter_isd_by_country_state(df, ctry, state)
    if subset.empty:
        return None

    enriched = add_geocoded_city_county_columns(
        subset, delay_sec=delay_sec, timeout_sec=timeout_sec
    )
    enriched.to_csv(dst_path, index=False)
    return dst_path
