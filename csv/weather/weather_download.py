"""
Download NOAA Global Hourly CSVs for stations listed in a geocoded ISD file.

Paths are resolved relative to this file's directory unless an absolute path is given.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from weather.csv_geocode import _float_lat_lon

NCEI_GLOBAL_HOURLY_ACCESS = "https://www.ncei.noaa.gov/data/global-hourly/access"

STATIONS_CSV_COLUMNS = (
    "name",
    "lat",
    "lon",
    "closest_station",
    "station_flags",
)

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


def _place_name_to_closest_station_csv(place_name: str) -> str:
    """Weather sidecar label: ``<name>`` for the place (city or county name)."""
    n = str(place_name).strip()
    return f"{n}" if n else ""


def append_stations_csv_row(
    stations_csv: Path,
    *,
    name: str,
    lat: float | None,
    lon: float | None,
) -> None:
    """
    Append one row to ``stations.csv`` (created next to downloaded weather CSVs).
    ``closest_station`` is ``<name>`` for the place label; ``station_flags`` is
    1 meaning that this is a city or county with a station
    """
    place = str(name).strip()
    row = {
        "name": place,
        "lat": lat if lat is not None else "",
        "lon": lon if lon is not None else "",
        "closest_station": _place_name_to_closest_station_csv(place),
        "station_flags": 1,
    }
    stations_csv.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row], columns=list(STATIONS_CSV_COLUMNS))
    if stations_csv.exists():
        existing = pd.read_csv(
            stations_csv,
            dtype={"name": str, "closest_station": str},
        )
        for col in STATIONS_CSV_COLUMNS:
            if col not in existing.columns:
                existing[col] = ""
        existing["closest_station"] = existing["closest_station"].astype(str).str.strip()
        existing_names = set(existing["name"].astype(str).str.strip())
        if place in existing_names:
            return
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["name"], keep="first")
        combined.to_csv(stations_csv, index=False)
    else:
        new_df.to_csv(stations_csv, index=False)


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
) -> tuple[bool, bool]:
    """
    Download one Global Hourly file. On failure, prints the required message.

    Returns:
        ``(ok, downloaded_new_file)`` where ``downloaded_new_file`` is True only
        when this call fetched and wrote the CSV (not when skipping or file existed).
    """
    url = global_hourly_csv_url(year, station_id, base=base_url)
    dest = out_dir
    if cities:
        dest = dest / f"{city}_{year}.csv"
    else:
        dest = dest / f"{county}_{year}.csv"

    if dest.exists():
        return True, False

    if county in sucess_counties and not cities:
        print(f"Skipping station {station_name} in {county} with station_id {station_id} for year {year} because county already has sucessful download.")
        return True, False

    if city in sucess_cities and cities:
        print(f"Skipping station {station_name} in {city} with station_id {station_id} for year {year} because city already has sucessful download.")
        return True, False

    ok = download_url_to_file(url, dest)
    if not ok:
        print(
            f"Station {station_name} in {county if not cities else city} with station_id {station_id} could not be Downloaded."
        )
        return False, False

    if county and county not in sucess_counties and not cities:
        sucess_counties.append(county)
    if city and city not in sucess_cities and cities:
        sucess_cities.append(city)
    return True, True


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
) -> bool:
    """
    Read ``src`` (default ``geocoded.csv`` next to this module), optionally restrict
    to rows whose ``Geocoded_County`` matches ``counties``, then for each distinct
    ``station_id`` attempt to download Global Hourly CSVs for each ``years``.

    ``cities``as a boolean will be a toggle. When false, only get one file per county, when true get one file per city.
    This allows for more flexible data collection.

    If a download fails (HTTP error, I/O, etc.), prints:
    ``Station <name> with station_id <id> could not be Downloaded.``

    After each successful new download (HTTP fetch in this run), appends one row to
    ``<out_dir>/stations.csv`` when that ``name`` is not already listed: ``name``
    (city or county label used for filenames), ``lat`` / ``lon`` from the deduped
    ISD row, ``closest_station`` (place label from ``_place_name_to_closest_station_csv``),
    and ``station_flags`` (unchanged from the module's registry format).
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
    stations_csv = out_path / "stations.csv"

    # To be used for logging
    total_files = 0
    stations_added = 0

    for year in years:
        # Reset sucess lists
        sucess_cities.clear()
        sucess_counties.clear()

        for _, row in deduped.iterrows():
            sid = str(row["station_id"])
            station_label = row.get("STATION NAME", "")
            county = row.get("Geocoded_County", "")
            city = row.get("Geocoded_City", "")
            if pd.isna(station_label):
                station_label = ""
            else:
                station_label = str(station_label)

            place_name = (
                ("" if pd.isna(city) else str(city).strip())
                if cities
                else ("" if pd.isna(county) else str(county).strip())
            )
            lat = _float_lat_lon(row.get("LAT"))
            lon = _float_lat_lon(row.get("LON"))

            ok, downloaded_new = download_station_year(
                cities,
                sid,
                station_label,
                year,
                out_path,
                base_url=base_url,
                county=county,
                city=city,
            )
            if ok and downloaded_new:
                append_stations_csv_row(
                    stations_csv,
                    name=place_name,
                    lat=lat,
                    lon=lon,
                )
                stations_added += 1


        print(f"Finished downloads for year {year}.")
        total_files +=  (len(sucess_cities) if cities else len(sucess_counties))
    
    if(stations_added == 0):
        print("No stations were added, catastrophic error, canceling data generation")
        return False

    if cities:
        print(f"Sucessfully downloaded weather data for {len(sucess_cities)} cities")
    else:
        print(f"Successfully downloaded weather data for {len(sucess_counties)} counties")
    print(f"Total files downloaded: {total_files}")

    return True
