"""
Summarize ISD station history for US Texas: count distinct locations after
normalizing names so e.g. "DALLAS, TX" and "DALLAS TX" map to the same key.

There is no separate City column in isd-history.csv; normalization is applied
to STATION NAME.

Reverse geocoding (city / county from LAT/LON) uses the free U.S. Census
coordinates geocoder. Run with: ``python isd-history-overview.py --geocode``

Count unique counties in the geocoded output:
``python isd-history-overview.py --count-counties``
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

CENSUS_COORD_GEOCODER = (
    "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
)


def normalize_tx_place(name: str) -> str:
    """
    Normalize a place string so trailing Texas markers collapse the same way:
    "Dallas, TX" / "DALLAS TX" / "Dallas,TX" -> "DALLAS"
    """
    if pd.isna(name):
        return ""
    s = str(name).strip().upper()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r",\s*TX\s*$", "", s)
    s = re.sub(r"\s+TX\s*$", "", s)
    return s.strip()


def write_unique_tx_cities_csv(
    history_csv_path: str,
    output_csv_path: str,
) -> str | None:
    """
    Read isd-history.csv, keep US / TX rows, normalize STATION NAME to city keys,
    and write one row per distinct city to output_csv_path.

    Returns:
        Path written, or None if no rows were produced.
    """
    df = pd.read_csv(history_csv_path)
    tx_us = df[(df["CTRY"] == "US") & (df["STATE"] == "TX")].copy()
    if tx_us.empty:
        return None

    tx_us["City"] = tx_us["STATION NAME"].map(normalize_tx_place)
    unique_cities = sorted({c for c in tx_us["City"] if c})
    out_df = pd.DataFrame({"City": unique_cities})
    out_df.to_csv(output_csv_path, index=False)
    print(
        f"Wrote {len(unique_cities)} rows to {output_csv_path}",
        flush=True,
    )
    return output_csv_path


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
    BASENAME when the point lies inside one; otherwise None. County is always
    set when the lookup succeeds. County subdivision may help interpret
    unincorporated locations (often a 'CCD' name).

    See: https://geocoding.geo.census.gov/geocoder/
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
        print(f"Geocoding row with {lat_col}={row.get(lat_col)} and {lon_col}={row.get(lon_col)}...", flush=True)
        lat = _float_lat_lon(row.get(lat_col))
        lon = _float_lat_lon(row.get(lon_col))
        if lat is None or lon is None:
            cities.append(None)
            counties.append(None)
            subdivs.append(None)
            continue

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


def write_tx_stations_geocoded_csv(
    history_csv_path: str,
    output_csv_path: str,
    *,
    delay_sec: float = 0.12,
) -> str | None:
    """
    Read isd-history.csv, keep US / TX rows, add Census city/county columns,
    and write output_csv_path.
    """
    df = pd.read_csv(history_csv_path)
    tx_us = df[(df["CTRY"] == "US") & (df["STATE"] == "TX")].copy()
    if tx_us.empty:
        return None

    enriched = add_geocoded_city_county_columns(tx_us, delay_sec=delay_sec)
    enriched.to_csv(output_csv_path, index=False)
    return output_csv_path


def count_unique_counties_in_geocoded_csv(geocoded_csv_path: str) -> int:
    """
    Count distinct non-empty ``Geocoded_County`` values in a CSV such as
    ``isd-history-tx-geocoded.csv`` (output of ``write_tx_stations_geocoded_csv``).

    Raises:
        ValueError: If the file has no ``Geocoded_County`` column.
        FileNotFoundError: If the path does not exist (from pandas read).
    """
    df = pd.read_csv(geocoded_csv_path)
    if "Geocoded_County" not in df.columns:
        raise ValueError(
            f"No 'Geocoded_County' column in {geocoded_csv_path!r} "
            "(run with --geocode first)."
        )
    counties = df["Geocoded_County"].dropna().astype(str).str.strip()
    counties = counties[counties != ""]
    return counties.nunique()


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "isd-history.csv")

    df = pd.read_csv(csv_path)
    tx_us = df[(df["CTRY"] == "US") & (df["STATE"] == "TX")].copy()

    tx_us["place_key"] = tx_us["STATION NAME"].map(normalize_tx_place)
    unique_places = tx_us["place_key"].nunique()
    row_count = len(tx_us)

    print(f"File: {csv_path}", flush=True)
    print(f"Rows with CTRY=US and STATE=TX: {row_count}", flush=True)
    print(
        "Distinct normalized STATION NAME values "
        '("Dallas, TX" / "Dallas TX" style merged): '
        f"{unique_places}",
        flush=True,
    )

    # Output is CSV (not .py), always next to this script: csv/weather/tx_unique_cities.csv
    cities_out = os.path.join(script_dir, "tx_unique_cities.csv")
    written = write_unique_tx_cities_csv(csv_path, cities_out)
    if written:
        print(f"Unique cities CSV: {written}", flush=True)
    else:
        print("No US/TX rows; skipped unique cities CSV.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if "--geocode" in sys.argv:
        src = os.path.join(script_dir, "isd-history.csv")
        dst = os.path.join(script_dir, "isd-history-tx-geocoded.csv")
        print(f"Census reverse geocoding US/TX rows -> {dst} ...")
        path = write_tx_stations_geocoded_csv(src, dst)
        if path:
            print(f"Done. Wrote {path}")
        else:
            print("No US/TX rows to geocode.")
    elif "--count-counties" in sys.argv:
        geo_path = os.path.join(script_dir, "isd-history-tx-geocoded.csv")
        if not os.path.isfile(geo_path):
            print(f"File not found: {geo_path}", flush=True)
            print("Run: py .\\isd-history-overview.py --geocode", flush=True)
            sys.exit(1)
        try:
            n = count_unique_counties_in_geocoded_csv(geo_path)
        except ValueError as e:
            print(e, flush=True)
            sys.exit(1)
        print(f"Unique counties in {geo_path}: {n}", flush=True)
    else:
        main()
