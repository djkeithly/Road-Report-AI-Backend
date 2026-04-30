import pandas as pd
from pathlib import Path

from csv_editing import _normalize_section_value

# Filename only: always written under this file's directory (not the process cwd).
output_csv_path = Path("RoadSections.csv")

_LAT_CANDIDATES = ("latitude", "Latitude", "LAT", "lat", "LATITUDE", "Y")
_LON_CANDIDATES = ("longitude", "Longitude", "LON", "lon", "LONGITUDE", "lng", "X")


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve_crash_data_path(crash_data_source: str | Path) -> str:
    """Absolute path: use as-is; relative paths are resolved against this file's directory."""
    p = Path(crash_data_source)
    if p.is_absolute():
        return str(p)
    return str(_script_dir() / p)


def _resolved_output_csv_path() -> Path:
    """Where ``output_csv_path`` is actually written (relative segment anchored to this module)."""
    return _script_dir() / output_csv_path


def create_section_data(input_csv):
    input_csv_path = _resolve_crash_data_path(input_csv)
    out_path = _resolved_output_csv_path()

    # Read the input CSV file
    df = pd.read_csv(input_csv_path)

    # Create a new DataFrame for the output
    output_df = pd.DataFrame(columns=["Street Name", "Section"])

    # Iterate through each row in the input DataFrame
    for index, row in df.iterrows():
        if pd.isna(row["Street Name"]) or pd.isna(row["Section"]) or row["Section"] == "No Data":
            continue  # Skip rows with missing values

        exists = not output_df[
            (output_df["Street Name"] == row["Street Name"])
            & (output_df["Section"] == row["Section"])
        ].empty
        if exists:
            continue  # Skip if this street + section is already in the output

        road_name = row["Street Name"]
        section = row["Section"]

        output_df = pd.concat(
            [output_df, pd.DataFrame([{"Street Name": road_name, "Section": section}])],
            ignore_index=True,
        )

    output_df.to_csv(out_path, index=False)
    try:
        display = out_path.relative_to(Path.cwd())
    except ValueError:
        display = out_path
    print(f"Road section data has been saved to {display}")


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def attach_average_coordinates(
    input_csv: str | Path,
    *,
    road_sections_csv: str | Path | None = None,
    latitude_column: str | None = None,
    longitude_column: str | None = None,
) -> Path:
    """
    For each (Street Name, Section) row in ``RoadSections.csv``, find all rows in
    ``input_csv`` with the same street and section (after the same section normalization
    as the rest of the pipeline), average their coordinates, and write ``latitude`` and
    ``longitude`` columns on the road-sections file.

    Latitude/longitude column names are taken from ``latitude_column`` / ``longitude_column``
    when provided; otherwise the first matching name from common variants is used.
    """
    crash_path = Path(_resolve_crash_data_path(input_csv))
    roads_path = (
        Path(road_sections_csv)
        if road_sections_csv is not None
        else _resolved_output_csv_path()
    )
    if not roads_path.is_file():
        raise FileNotFoundError(f"Road sections file not found: {roads_path}")
    if not crash_path.is_file():
        raise FileNotFoundError(f"Input crash CSV not found: {crash_path}")

    roads = pd.read_csv(roads_path)
    crash = pd.read_csv(crash_path)

    for col in ("Street Name", "Section"):
        if col not in roads.columns:
            raise ValueError(f"Road sections CSV missing required column {col!r}: {roads_path}")
        if col not in crash.columns:
            raise ValueError(f"Input CSV missing required column {col!r}: {crash_path}")

    lat_col = latitude_column or _pick_column(crash, _LAT_CANDIDATES)
    lon_col = longitude_column or _pick_column(crash, _LON_CANDIDATES)
    if lat_col is None or lon_col is None:
        raise ValueError(
            f"Could not find latitude/longitude columns in {crash_path}. "
            f"Columns present: {list(crash.columns)}. "
            "Pass latitude_column= and longitude_column= explicitly."
        )

    c = crash.copy()
    c["_street_key"] = c["Street Name"].map(
        lambda x: str(x).strip().upper() if pd.notna(x) else ""
    )
    c["_section_key"] = c["Section"].map(_normalize_section_value).map(lambda x: str(x).upper())

    valid_coord = c[lat_col].notna() & c[lon_col].notna()
    c["_lat"] = pd.to_numeric(c[lat_col], errors="coerce")
    c["_lon"] = pd.to_numeric(c[lon_col], errors="coerce")
    valid_coord &= c["_lat"].notna() & c["_lon"].notna()

    agg = (
        c.loc[valid_coord]
        .groupby(["_street_key", "_section_key"], dropna=False)
        .agg(latitude=("_lat", "mean"), longitude=("_lon", "mean"))
        .reset_index()
    )

    out = roads.copy()
    out["_street_key"] = out["Street Name"].map(
        lambda x: str(x).strip().upper() if pd.notna(x) else ""
    )
    out["_section_key"] = out["Section"].map(_normalize_section_value).map(lambda x: str(x).upper())

    out = out.merge(agg, on=["_street_key", "_section_key"], how="left")
    out = out.drop(columns=["_street_key", "_section_key"])

    out.to_csv(roads_path, index=False)
    try:
        display = roads_path.relative_to(Path.cwd())
    except ValueError:
        display = roads_path
    print(f"Updated road sections with mean coordinates: {display}")
    return roads_path
