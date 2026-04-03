"""
Download NOAA Global Hourly CSVs for stations listed in a geocoded ISD file.

Paths are resolved relative to this file's directory unless an absolute path is given.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

NCEI_GLOBAL_HOURLY_ACCESS = "https://www.ncei.noaa.gov/data/global-hourly/access"

sucess_cities = []
sucess_counties = []

def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def format_station_id(usaf, wban) -> str:
    """ISD composite id: 6-digit USAF + 5-digit WBAN (zero-padded)."""
    u = str(int(float(usaf))).zfill(6)
    w = str(int(float(wban))).zfill(5)
    return u + w


def row_matches_county_filter(geocoded_county, county_tokens: list[str]) -> bool:
    """
    True if any token appears in the county string (case-insensitive).
    E.g. token ``Dallas`` matches ``Dallas County``.
    """
    if pd.isna(geocoded_county):
        return False
    gc = str(geocoded_county).strip().lower()
    for raw in county_tokens:
        token = raw.strip().lower()
        if not token:
            continue
        if token in gc:
            return True
    return False


def filter_stations_by_ctry_state(df: pd.DataFrame, ctry: str | None, state: str | None) -> pd.DataFrame:
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
    
    if ctry is not None:
        return df[df["CTRY"] == ctry].copy()

    if state is not None:
        return df[df["STATE"] == state].copy()

    return df[df["STATE"] == state].copy()

def filter_stations_by_counties(df: pd.DataFrame, counties: list[str] | None) -> pd.DataFrame:
    """
    Keep rows whose ``Geocoded_County`` matches any string in ``counties``.
    If ``counties`` is None or empty, returns ``df`` unchanged (all rows).
    """
    if not counties:
        return df

    if "Geocoded_County" not in df.columns:
        raise ValueError("geocoded CSV must include a 'Geocoded_County' column when counties is set.")

    mask = df["Geocoded_County"].apply(lambda gc: row_matches_county_filter(gc, counties))
    return df.loc[mask].copy()


def load_geocoded_with_station_ids(src: Path) -> pd.DataFrame:
    """Read geocoded CSV and add ``station_id`` column."""
    df = pd.read_csv(src)
    usaf = pd.to_numeric(df["USAF"], errors="coerce")
    wban = pd.to_numeric(df["WBAN"], errors="coerce")
    df["station_id"] = (
        usaf.fillna(0).astype(int).astype(str).str.zfill(6)
        + wban.fillna(0).astype(int).astype(str).str.zfill(5)
    )
    return df


def global_hourly_csv_url(year: int, station_id: str, base: str = NCEI_GLOBAL_HOURLY_ACCESS) -> str:
    return f"{base.rstrip('/')}/{year}/{station_id}.csv"


def download_url_to_file(url: str, dest: Path) -> bool:
    """Download full response body to ``dest``. Returns True on HTTP 200 and success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "RoadReportWeather/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            if getattr(resp, "status", 200) != 200:
                return False
            body = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return False

    try:
        dest.write_bytes(body)
    except OSError:
        return False
    return True


def download_station_year(
    cities: bool,
    station_id: str,
    station_name: str,
    year: int,
    out_dir: Path,
    *,
    base_url: str = NCEI_GLOBAL_HOURLY_ACCESS,
    county: str | None = None,
    city: str | None = None,
) -> bool:
    """
    Download one Global Hourly file. On failure, prints the required message and returns False.
    On skip_if_exists and file present, returns True without downloading.
    """
    url = global_hourly_csv_url(year, station_id, base=base_url)
    dest = out_dir
    if cities:
        dest = dest / f"{city}_{year}.csv"
    else:
        dest = dest / f"{county}_{year}.csv"

    if dest.exists():
        return True
    
    if county in sucess_counties and not cities:
        print(f"Skipping station {station_name} in {county} with station_id {station_id} for year {year} because county already has sucessful download.")
        return True
    
    if city in sucess_cities and cities:
        print(f"Skipping station {station_name} in {city} with station_id {station_id} for year {year} because city already has sucessful download.")
        return True

    ok = download_url_to_file(url, dest)
    if not ok:
        print(
            f"Station {station_name} in {county if not cities else city} with station_id {station_id} could not be Downloaded."
        )
    else:
        if county and county not in sucess_counties and not cities:
            sucess_counties.append(county)
        if city and city not in sucess_cities and cities:
            sucess_cities.append(city)
    return ok


def create_weather_files(
    src: str = "geocoded.csv",
    ctry: str = "US",
    state: str = "TX",
    counties: list[str] | None = None,
    cities: bool = False,
    *,
    years: list[int] | None = None,
    out_dir: str | Path = "weather_csvs",
    base_url: str = NCEI_GLOBAL_HOURLY_ACCESS,
) -> None:
    """
    Read ``src`` (default ``geocoded.csv`` next to this module), optionally restrict
    to rows whose ``Geocoded_County`` matches ``counties``, then for each distinct
    ``station_id`` attempt to download Global Hourly CSVs for each ``years``.

    ``cities``as a boolean will be a toggle. When false, only get one file per county, when true get one file per city.
    This allows for more flexible data collection.

    If a download fails (HTTP error, I/O, etc.), prints:
    ``Station <name> with station_id <id> could not be Downloaded.``
    """

    years = years if years is not None else [2025]
    root = _script_dir()
    src_path = Path(src) if Path(src).is_absolute() else root / src
    out_path = Path(out_dir) if Path(out_dir).is_absolute() else root / out_dir

    if(state is not None and ctry is not None and ctry != "US"):
        raise ValueError("Currently only US country codes are supported when state is specified.")

    df = load_geocoded_with_station_ids(src_path)
    df = filter_stations_by_ctry_state(df, ctry=ctry, state=state)
    df = filter_stations_by_counties(df, counties=counties)

    if df.empty:
        return

    # One download per station_id per year (first row supplies the station name for messages).
    deduped = df.drop_duplicates(subset=["station_id"], keep="first")

    # To be used for logging
    total_files = 0

    for year in years:
        # Reset sucess lists
        sucess_cities.clear()
        sucess_counties.clear()

        for _, row in deduped.iterrows():
            sid = str(row["station_id"])
            name = row.get("STATION NAME", "")
            county = row.get("Geocoded_County", "")
            city = row.get("Geocoded_City", "")
            if pd.isna(name):
                name = ""
            else:
                name = str(name)

            download_station_year(
                cities,
                sid,
                name,
                year,
                out_path,
                base_url=base_url,
                county=county,
                city=city,
            )

        print(f"Finished downloads for year {year}.")
        total_files +=  (len(sucess_cities) if cities else len(sucess_counties))
    
    if cities:
        print(f"Sucessfully downloaded weather data for {len(sucess_cities)} cities")
    else:
        print(f"Successfully downloaded weather data for {len(sucess_counties)} counties")
    print(f"Total files downloaded: {total_files}")
