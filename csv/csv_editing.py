## This file will create negative samples for the model by creating new rows that represnent non-crash events.
# It will use the existing data to create these samples, ensuring that they are realistic and varied.
#
# Importable entry point: ``training_data_production(crash_data_source)``.

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from datetime import datetime, timedelta
import math
import re

#   #   #   #   #   #   #   #   #   #   #   #
#               Step 1 Functions            #
#   #   #   #   #   #   #   #   #   #   #   #

# A basic function that takes a CSV file and adds a row called Crash and sets all the values in the file to 1, indicating a crash
# This allows for negative sampling, or the ability to differentiate between crash and non crash events in the dataset
# This will be critical to actually training an AI model
def _extract_hour_bucket(value):
    """Return hour as string without leading zeros (e.g., '01:00 - 01:59' -> '1')."""
    if pd.isna(value):
        return value

    text = str(value).strip()
    match = re.match(r"^(\d{1,2})", text)
    if not match:
        return text

    return str(int(match.group(1)))


def _normalize_section_value(val) -> str:
    """Section for grouping and keys: empty string means no section / not applicable."""
    if pd.isna(val):
        return ""
    if isinstance(val, bool):
        return str(val).strip()
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return ""
        if isinstance(val, float) and val == int(val):
            return str(int(val))
        if isinstance(val, int):
            return str(val)
        return str(val).strip()
    else:
        s = str(val).strip()
    if s.lower() in ("no data", "nan", ""):
        return ""
    try:
        f = float(s)
        if not math.isnan(f) and f == int(f):
            return str(int(f))
    except ValueError:
        pass
    return s


def _event_key_street_section_date_hour(row) -> tuple:
    """Crash/negative dedupe key: (street upper, normalized section, date, hour bucket)."""
    sn = row["Street Name"] if "Street Name" in row.index else None
    street = str(sn).strip().upper() if sn is not None and pd.notna(sn) else ""
    if "Section" in row.index:
        sec = _normalize_section_value(row["Section"]).upper()
    else:
        sec = ""
    hour = _extract_hour_bucket(row["Hour of Day"])
    return (street, sec, row["Crash Date"], hour)


def add_crash_column(filepath):
    """
    Add a 'Crash' column to the CSV file with all values set to 1.
    This indicates that all rows in the dataset represent crash events.
    Removes a 'Crash ID' column if present.

    Args:
        filepath (str): Path to the input CSV file
        output_filename (str): Optional custom output filename/path.
            If None, the input file is updated in place.

    Returns:
        dict: Dictionary containing:
            - 'total_rows': Number of rows processed
            - 'output_file': Path to the updated CSV file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)

        # Clean up step
        df = df.drop(columns=["Crash ID", "Average Daily Traffic Amount"], errors="ignore")
        if "Hour of Day" in df.columns:
            df["Hour of Day"] = df["Hour of Day"].apply(_extract_hour_bucket)

        # Add the 'Crash' column with all values set to 1
        df['Crash'] = 1
        
        # Determine output file path
        # If no output is provided, modify the original file in place.            # Preserve existing behavior for bare filenames.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.isabs(filepath):
            output_file = filepath
        else:
            output_file = os.path.join(script_dir, filepath)

        # Save the modified CSV
        df.to_csv(output_file, index=False)
        
        return {
            'total_rows': len(df),
            'output_file': output_file
        }
    
    except Exception as e:
        print(f"Error adding 'Crash' column: {e}")
        return None

#   #   #   #   #   #   #   #   #   #   #   #
#               Step 2 Functions            #
#   #   #   #   #   #   #   #   #   #   #   #

def combine_street_name_and_segment(filepath):
    """
    Combine 'Street Name' and 'Section' into a single 'Street Name' column
    in the format '<Street Name> <Section>', then drop the 'Section' column.
    If 'Section' is 'No Data', the street name is left unchanged for that row.

    Args:
        filepath (str): Path to the input CSV file.
    """
    try:
        df = pd.read_csv(filepath)

        if "Street Name" in df.columns and "Section" in df.columns:
            # Only append section where it has meaningful data
            mask = df["Section"].astype(str) != "No Data"
            df.loc[mask, "Street Name"] = (
                df.loc[mask, "Street Name"].astype(str)
                + " S"
                + df.loc[mask, "Section"].astype(str)
            )
            df = df.drop(columns=["Section"])

            script_dir = os.path.dirname(os.path.abspath(__file__))
            if os.path.isabs(filepath):
                output_file = filepath
            else:
                output_file = os.path.join(script_dir, filepath)

            df.to_csv(output_file, index=False)
        else:
            print("Columns 'Street Name' and/or 'Section' not found; no changes made.")

    except Exception as e:
        print(f"Error combining 'Street Name' and 'Section': {e}")

#   #   #   #   #   #   #   #   #   #   #   #
#               Step 3 Functions            #
#   #   #   #   #   #   #   #   #   #   #   #

def get_average_crashes_per_road(filepath):
    """
    Calculate the average number of crashes per road (or per road + section when
    ``Section`` is present) in the dataset.
    
    Args:
        filepath (str): Path to the input CSV file
        
    Returns:
        int: Average number of crashes per road rounded up to the nearest whole number
    """
    try:
        df = pd.read_csv(filepath)
        if "Section" in df.columns:
            valid = df["Street Name"].notna()
            sec_norm = df["Section"].map(_normalize_section_value)
            road_counts = (
                df.loc[valid]
                .assign(_sec_norm=sec_norm.loc[valid])
                .groupby(["Street Name", "_sec_norm"], dropna=False)
                .size()
            )
        else:
            road_counts = df.loc[df["Street Name"].notna(), "Street Name"].value_counts()
        average_crashes = math.ceil(road_counts.mean())
        return average_crashes
    except Exception as e:
        print(f"Error calculating average crashes per road: {e}")
        return None

# This file looks through the data set and identifies roads that are non-critical
# For the specific example of Dallas County, this is set any roads with less than 12 crashes
# It then creates a modified CSV where those roads are renamed to 'Non Critical Road'
def identify_and_replace_non_critical_roads(filepath, crash_threshold=12):
    """
    Identify roads with less than the crash threshold and create a modified CSV
    where those roads are renamed to 'Non Critical Road'.
    
    Args:
        filepath (str): Path to the input CSV file
        crash_threshold (int): Minimum number of crashes to be considered "critical"
        
    Returns:
        dict: Dictionary containing:
            - 'non_critical_roads': List of roads with less than threshold crashes
            - 'count_non_critical': Number of roads identified as non-critical
            - 'output_file': Path to the new modified CSV file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)

        df_modified = df.copy()
        if "Section" in df_modified.columns:
            # Rows with missing street name must be excluded from grouping: a (nan, sec)
            # MultiIndex key breaks ``road_counts[key]`` lookups (NaN != NaN).
            valid_street = df_modified["Street Name"].notna()
            sec_norm = df_modified["Section"].map(_normalize_section_value)
            road_counts = (
                df_modified.loc[valid_street]
                .assign(_sec_norm=sec_norm.loc[valid_street])
                .groupby(["Street Name", "_sec_norm"], dropna=False)
                .size()
            )
            non_critical_pairs = road_counts[road_counts < crash_threshold].index.tolist()
            # One vectorized pass: avoid per-pair full-column ``.map(_normalize_section_value)``
            # (that was O(len(non_critical_pairs) * n_rows) and dominated runtime).
            if not non_critical_pairs:
                mask = pd.Series(False, index=df_modified.index)
            else:
                pair_idx = pd.MultiIndex.from_arrays(
                    [df_modified["Street Name"].to_numpy(), sec_norm.to_numpy()],
                )
                nc_idx = pd.MultiIndex.from_tuples(non_critical_pairs)
                mask = pd.Series(
                    valid_street.to_numpy() & pair_idx.isin(nc_idx),
                    index=df_modified.index,
                )
            df_modified.loc[mask, "Street Name"] = "Non Critical Road"
            df_modified.loc[mask, "Section"] = "No Data"
            non_critical_roads = [f"{n} | sec={s!r}" for n, s in non_critical_pairs]
        else:
            road_counts = df["Street Name"].value_counts()
            non_critical_roads = road_counts[road_counts < crash_threshold].index.tolist()
            df_modified.loc[df_modified["Street Name"].isin(non_critical_roads), "Street Name"] = (
                "Non Critical Road"
            )
        
        # Get the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(filepath))[0]}_modified.csv")
        
        # Save the modified CSV
        df_modified.to_csv(output_file, index=False)
        
        # Sort and display non-critical roads (or road+section pairs) with crash counts
        if "Section" in df_modified.columns:
            count_keys = non_critical_pairs
        else:
            count_keys = non_critical_roads
        non_critical_with_counts = [(k, road_counts[k]) for k in count_keys]
        non_critical_with_counts.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'non_critical_roads': non_critical_roads,
            'count_non_critical': len(non_critical_roads),
            'output_file': output_file,
            'non_critical_with_counts': non_critical_with_counts
        }
        
    except Exception as e:
        print(f"Error processing roads: {e}")
        return None
    
#   #   #   #   #   #   #   #   #   #   #   #
#               Step 4 Functions            #
#   #   #   #   #   #   #   #   #   #   #   #

# This file creates a general reference CSV with information about each type of unique road in the dataset
# This allows the above function, identify_and_replace_non_critical_roads, to determine what roads are non critical
# It also allows the negative sampling function, create_negative_samples, to create realistic negative samples by using actual road attributes from the dataset.
def create_road_reference_csv(filepath):
    """
    Create a CSV file with unique road **segments** from the modified dataset: each row is
    a ``(Road Name, Section)`` pair (Section replaces the former rural/urban column in the
    reference schema). Rows without a meaningful section share one bucket (empty / ``No Data``).
    Used for negative sampling keyed by name + section.
    
    Args:
        filepath (str): Path to the modified CSV file
        
    Returns:
        dict: Dictionary containing:
            - 'unique_roads': Number of unique road segments found
            - 'output_file': Path to the created reference file
            - 'road_list': List of road dictionaries
    """
    try:
        # Read the modified CSV file
        df = pd.read_csv(filepath)
        if "Section" not in df.columns:
            df = df.copy()
            df["Section"] = ""

        df = df.copy()
        df = df.loc[df["Street Name"].notna()].copy()
        df["_section_key"] = df["Section"].map(_normalize_section_value)

        road_reference = (
            df.groupby(["Street Name", "_section_key"], dropna=False)
            .agg(
                {
                    "Section": "first",
                    "County": "first",
                    "City": lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
                }
            )
            .reset_index()
            .drop(columns=["_section_key"])
        )

        road_reference.rename(columns={"Street Name": "Road Name"}, inplace=True)
        road_reference = road_reference[["Road Name", "Section", "County", "City"]]
        road_reference = road_reference.sort_values(["Road Name", "Section"]).reset_index(drop=True)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, "RoadReference.csv")

        road_reference.to_csv(output_file, index=False)

        section_counts = road_reference["Section"].fillna("").astype(str).value_counts()

        return {
            "unique_roads": len(road_reference),
            "output_file": output_file,
            "road_list": road_reference.to_dict("records"),
            "section_value_counts": section_counts,
        }

    except Exception as e:
        print(f"Error creating road reference CSV: {e}")
        return None

#   #   #   #   #   #   #   #   #   #   #   #
#               Step 5 Functions            #
#   #   #   #   #   #   #   #   #   #   #   #

# This will loop through all the modified roads (Non Critical Road) and create negative samples for each road, each hour, and each day of the year
# This function will take a few minutes to run and will update the user of its progress in the console log
# Out put is negative_samples.csv
_CITY_WEATHER_LOOKUP_CACHE: dict[tuple[str, int], dict[tuple[str, str], tuple[str, str]]] = {}


def _safe_name_for_file(name: str) -> str:
    """Match ``weather/open_download.py`` naming for ``weather_csvs/<stem>_<year>.csv``."""
    s = str(name).strip()
    for c in r'<>:"/\|?*':
        s = s.replace(c, "_")
    return s or "place"


def _hour_key_from_city_csv_hour(val: object) -> str | None:
    if pd.isna(val):
        return None
    try:
        return f"{int(float(val)):02d}"
    except (TypeError, ValueError):
        return None


def _read_city_weather_lookup(weather_csv_path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """
    Build (date ``YYYY-MM-DD``, hour ``00``..``23``) -> (weather label, road label)
    from an Open-Meteo export CSV (3-line preamble + header with ``date``, ``hour``,
    ``weather``, ``Road``).
    """
    if not weather_csv_path.is_file():
        return {}
    df = pd.read_csv(weather_csv_path, skiprows=3)
    colmap = {str(c).lower(): c for c in df.columns}
    for need in ("date", "hour", "weather", "road"):
        if need not in colmap:
            print(
                f"Warning: {weather_csv_path.name} missing column like '{need}' "
                f"(have: {list(df.columns)}); skipping file."
            )
            return {}
    dcol, hcol, wcol, rcol = colmap["date"], colmap["hour"], colmap["weather"], colmap["road"]
    out: dict[tuple[str, str], tuple[str, str]] = {}
    for _, row in df.iterrows():
        ds = str(row[dcol]).strip()
        hk = _hour_key_from_city_csv_hour(row[hcol])
        if hk is None:
            continue
        key = (ds, hk)
        if key not in out:
            out[key] = (str(row[wcol]).strip(), str(row[rcol]).strip())
    return out


def _get_city_weather_lookup(city: object, year: int) -> dict[tuple[str, str], tuple[str, str]]:
    if city is None or (isinstance(city, float) and pd.isna(city)):
        return {}
    s = str(city).strip()
    if not s:
        return {}
    stem = _safe_name_for_file(s)
    ck = (stem, year)
    if ck not in _CITY_WEATHER_LOOKUP_CACHE:
        path = (
            Path(__file__).resolve().parent
            / "weather"
            / "weather_csvs"
            / f"{stem}_{year}.csv"
        )
        _CITY_WEATHER_LOOKUP_CACHE[ck] = _read_city_weather_lookup(path)
    return _CITY_WEATHER_LOOKUP_CACHE[ck]


def create_negative_samples(
    year,
    road_reference_file="RoadReference.csv",
    final_weather_file: str | None = "FinalWeather.csv",
    output_filename="NegativeSamples.csv",
):
    """
    Create negative samples (non-crash events) for every **road segment** (``Road Name`` +
    ``Section`` from ``RoadReference.csv``), every hour, every day of the year.

    Weather and surface (road) conditions come from ``weather/weather_csvs/<City>_<year>.csv``
    for each road's ``City`` (same stem rules as ``open_download``), keyed by calendar date
    and hour. If that file or key is missing, falls back to ``final_weather_file`` when it
    exists (Date / Hour / Weather / Road), then to clear / dry defaults.

    Args:
        road_reference_file: Path to the road reference CSV (columns ``Road Name``, ``Section``, …).
        output_filename: Name for the output file in the script directory.
        final_weather_file: Optional legacy global lookup CSV; pass ``None`` to disable.

    Returns:
        dict: Summary statistics about the generated data
    """
    try:
        global _CITY_WEATHER_LOOKUP_CACHE
        _CITY_WEATHER_LOOKUP_CACHE.clear()

        script_dir = os.path.dirname(os.path.abspath(__file__))

        road_reference_path = road_reference_file
        if not os.path.isabs(road_reference_path):
            road_reference_path = os.path.join(script_dir, road_reference_path)

        weather_file_path = final_weather_file
        if weather_file_path and not os.path.isabs(weather_file_path):
            weather_file_path = os.path.join(script_dir, weather_file_path)

        road_df = pd.read_csv(road_reference_path)

        global_lookup: dict[tuple[str, str], tuple[str, str]] = {}
        if weather_file_path and os.path.exists(weather_file_path):
            global_lookup = _build_weather_lookup(weather_file_path)
            print(
                f"Optional global weather lookup: {len(global_lookup)} entries from {weather_file_path}."
            )
        elif final_weather_file:
            print(
                f"No FinalWeather at {weather_file_path}; "
                "using only per-city files under weather/weather_csvs/ (and defaults)."
            )

        cities = road_df["City"].dropna().astype(str).unique().tolist()
        cities_with_files = sum(1 for c in cities if len(_get_city_weather_lookup(c, year)) > 0)
        print(
            f"Per-city Open-Meteo CSVs ({year}): {cities_with_files}/{len(cities)} "
            f"distinct cities have data under weather/weather_csvs/"
        )

        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59, 59)

        rows = []
        crash_id_counter = 30000000

        print("\n" + "=" * 70)
        print("GENERATING NEGATIVE SAMPLES")
        print("=" * 70)
        print(f"Total road segments (name + section): {len(road_df)}")
        print("Generating records for every hour of the year...")
        print("-" * 70)

        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            day_name = current_date.strftime("%A").upper()
            if current_date.day == 1 and current_date.hour == 0:
                print(
                    f"Processing Month: {current_date.strftime('%B %Y')} ({date_str}) - Day: {day_name}"
                )

            for hour in range(24):
                hour_key = f"{hour:02d}"
                hour_range = f"{hour:02d}:00 - {hour:02d}:59"

                for _, road in road_df.iterrows():
                    city_lookup = _get_city_weather_lookup(road["City"], year)
                    weather_condition, surface_condition = city_lookup.get(
                        (date_str, hour_key),
                        global_lookup.get(
                            (date_str, hour_key),
                            ("1 - CLEAR", "1 - DRY"),
                        ),
                    )
                    sec = road["Section"] if "Section" in road_df.columns else ""
                    if pd.isna(sec):
                        sec = ""
                    rows.append(
                        {
                            "City": road["City"],
                            "County": road["County"],
                            "Crash Date": date_str,
                            "Crash Month": current_date.month,
                            "Crash Time": f"{hour:02d}00",
                            "Crash Year": year,
                            "Day of Week": day_name,
                            "Hour of Day": hour_range,
                            "Road Class": "CITY STREET",
                            "Street Name": road["Road Name"],
                            "Section": sec,
                            "Surface Condition": surface_condition,
                            "Weather Condition": weather_condition,
                            "Crash": 0,
                        }
                    )
                    crash_id_counter += 1

            current_date += timedelta(days=1)

        negative_df = pd.DataFrame(rows)

        output_path = os.path.join(script_dir, output_filename)
        negative_df.to_csv(output_path, index=False)

        print(f"[OK] Negative samples saved: {output_path}")
        print(f"[OK] Total records: {len(negative_df):,}")

        return {"total_records": len(negative_df), "output_file": output_path}

    except Exception as e:
        print(f"Error creating negative samples: {e}")
        return None
    
def _hour_key_from_range(hour_text: str) -> str | None:
    # Handles "13:00-13:59" and "13:00 - 13:59"
    if pd.isna(hour_text):
        return None
    s = str(hour_text).strip().replace(" ", "")
    return s.split(":")[0] if ":" in s else None

def _build_weather_lookup(final_weather_path: str) -> dict[tuple[str, str], tuple[str, str]]:
    """
    Returns lookup:
      (date_str, hour_key) -> (weather_condition, surface_condition)
    Uses the most frequent Weather/Road pair if duplicates exist in FinalWeather.
    """
    if not final_weather_path or not os.path.exists(final_weather_path):
        return {}

    wdf = pd.read_csv(final_weather_path)

    required_cols = {"Date", "Hour", "Weather", "Road"}
    missing = required_cols - set(wdf.columns)
    if missing:
        raise ValueError(f"FinalWeather.csv missing columns: {sorted(missing)}")

    wdf["Date"] = wdf["Date"].astype(str).str.strip()
    wdf["hour_key"] = wdf["Hour"].apply(_hour_key_from_range)
    wdf["Weather"] = wdf["Weather"].astype(str).str.strip()
    wdf["Road"] = wdf["Road"].astype(str).str.strip()

    wdf = wdf.dropna(subset=["Date", "hour_key", "Weather", "Road"])

    combo_counts = (
        wdf.groupby(["Date", "hour_key", "Weather", "Road"])
        .size()
        .reset_index(name="count")
        .sort_values(["Date", "hour_key", "count"], ascending=[True, True, False])
    )

    lookup = {}
    for _, row in combo_counts.iterrows():
        key = (row["Date"], row["hour_key"])
        if key not in lookup:
            lookup[key] = (row["Weather"], row["Road"])

    return lookup


#   #   #   #   #   #   #   #   #   #   #   #
#               Step 5 Functions            #
#   #   #   #   #   #   #   #   #   #   #   #

# This function takes the samples in negative_samples.csv and combines them with the crash samples in the modified csv with the crash row
# This will runs under the assumption that one car passes on any given road at least once an hour
# IF there is a crash on a given road at a given hour, the non crash sample for that road and hour will be removed to avoid conflicting data
# This should provide a reasonable scope of crashes and non crashes
def combine_crash_and_negative_samples(crash_file, negative_file="NegativeSamples.csv", output_filename="TrainingData.csv"):
    """
    Combine positive crash samples with negative samples, removing any negative samples
    that conflict with actual crash data (same street name, section when present, date, and hour).
    
    Args:
        crash_file (str): Path to the crash labeled CSV file
        negative_file (str): Path to the negative samples CSV file
        output_filename (str): Name for the output training data file
        
    Returns:
        dict: Summary statistics about the combined dataset
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        crash_path = crash_file if os.path.isabs(crash_file) else os.path.join(script_dir, crash_file)
        negative_path = negative_file if os.path.isabs(negative_file) else os.path.join(script_dir, negative_file)

        if not os.path.exists(crash_path):
            raise FileNotFoundError(
                f"Crash file not found: {crash_file} (resolved path: {crash_path})"
            )

        if not os.path.exists(negative_path):
            raise FileNotFoundError(
                f"Negative samples file not found: {negative_file} (resolved path: {negative_path})"
            )

        # Read both CSV files
        print("Reading crash data...")
        crash_df = pd.read_csv(crash_path)

        print("Reading negative samples...")
        negative_df = pd.read_csv(negative_path)

        # Create a set of crash events (road, date, hour) for fast lookup
        crash_events = set()

        print("Building crash event index...")
        for _, row in crash_df.iterrows():
            if pd.isna(row["Street Name"]):
                continue
            crash_events.add(_event_key_street_section_date_hour(row))
        
        print(f"Total crash events indexed: {len(crash_events)}")
        print("-" * 70)
        
        # Filter negative samples
        print("Filtering negative samples...")
        filtered_negative = []
        skipped_count = 0
        
        for _, row in negative_df.iterrows():
            if pd.isna(row["Street Name"]):
                continue

            event_key = _event_key_street_section_date_hour(row)

            # Only add if there's no conflicting crash event
            if event_key not in crash_events:
                filtered_negative.append(row)
            else:
                skipped_count += 1
        
        print(f"Negative samples kept: {len(filtered_negative)}")
        print(f"Negative samples skipped (conflicts): {skipped_count}")
        print("-" * 70)
        
        # Create DataFrames
        crash_df_copy = crash_df.copy()
        negative_df_filtered = pd.DataFrame(filtered_negative)
        
        # Combine the datasets
        print("Combining datasets...")
        training_data = pd.concat([crash_df_copy, negative_df_filtered], ignore_index=True)
        
        # Sort by date and time for better organization
        training_data = training_data.sort_values(['Crash Date', 'Hour of Day']).reset_index(drop=True)
        
        # Save to CSV
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, output_filename)
        training_data.to_csv(output_file, index=False)
        
        # Print summary
        print("-" * 70)
        print(f"\n[OK] Training data created successfully!")
        print(f"  Crash samples: {len(crash_df):,}")
        print(f"  Negative samples (filtered): {len(negative_df_filtered):,}")
        print(f"  Total training records: {len(training_data):,}")
        if len(crash_df) > 0:
            print(f"  Crash/Negative ratio: 1:{len(negative_df_filtered) // len(crash_df)}")
        print(f"  Output file: {output_file}")
        print("=" * 70)
        
        return {
            'crash_samples': len(crash_df),
            'negative_samples': len(negative_df_filtered),
            'total_records': len(training_data),
            'skipped_conflicts': skipped_count,
            'output_file': output_file
        }
        
    except Exception as e:
        print(f"Error combining datasets: {e}")
        import traceback
        traceback.print_exc()
        return None

def _abs_in_script_dir(filename: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)


def _resolve_crash_data_path(crash_data_source: str | Path) -> str:
    """Absolute path: use as-is; relative paths are resolved against this file's directory."""
    p = Path(crash_data_source)
    if p.is_absolute():
        return str(p)
    return str(Path(__file__).resolve().parent / p)


def training_data_production(crash_data_source: str | Path) -> dict[str, object] | None:
    """
    Run the full crash CSV → training dataset pipeline for a single crash export.

    Steps (in order): add ``Crash`` column; combine street + section (optional); replace non-critical
    roads using mean crashes per road or per (road, section) when ``Section`` exists (rounded up);
    build ``RoadReference.csv``;
    build ``NegativeSamples.csv`` (weather from ``weather/weather_csvs/<City>_<year>.csv``,
    optional ``FinalWeather.csv`` fallback);
    merge into ``TrainingData.csv``.

    Intermediate files (``*_modified.csv``, ``RoadReference.csv``, ``NegativeSamples.csv``)
    are written in the same directory as this script, matching ``identify_and_replace_non_critical_roads``.

    Args:
        crash_data_source: Path to the raw crash CSV (e.g. ``Dallas2025.csv`` relative to
            this file's folder, or an absolute path). The file is updated in place for the
            first two steps.

    Returns:
        A dict with paths and counts from each stage, or ``None`` if a step fails
        (e.g. missing weather file for negative sampling).
    """
    crash_path = _resolve_crash_data_path(crash_data_source)
    if not os.path.isfile(crash_path):
        raise FileNotFoundError(f"Crash data file not found: {crash_path}")

    summary: dict[str, object] = {"crash_source": crash_path}

    file_year = int(pd.read_csv(crash_path, nrows=1)["Crash Year"].iloc[0])
    summary["crash_year"] = file_year

    print("Task [1/6]: Adding crash column...")
    r1 = add_crash_column(crash_path)
    if r1 is None:
        print("Failed at add_crash_column.")
        return None
    summary["add_crash_column"] = r1

    average_crashes = get_average_crashes_per_road(crash_path)
    if average_crashes is None:
        print("Failed: could not compute average crashes per road.")
        return None
    summary["average_crashes_per_road_threshold"] = average_crashes

    # print("Task [2/6]: Combining street name and segment...")
    # combine_street_name_and_segment(crash_path)
    # print("Street name and segment combined.\n")

    print("Task [3/6]: Removing non-critical roads and creating modified CSV...")
    nc = identify_and_replace_non_critical_roads(crash_path, crash_threshold=average_crashes)
    if nc is None:
        print("Failed at identify_and_replace_non_critical_roads.")
        return None
    summary["non_critical_roads"] = nc
    modified_file = nc["output_file"]

    print("Task [4/6]: Creating road reference CSV...")
    rr = create_road_reference_csv(modified_file)
    if rr is None:
        print("Failed at create_road_reference_csv.")
        return None
    summary["road_reference"] = rr

    print("Task [5/6]: Creating negative samples...")
    neg = create_negative_samples(
        file_year,
        road_reference_file="RoadReference.csv",
        final_weather_file="FinalWeather.csv",
        output_filename="NegativeSamples.csv",
    )
    if neg is None:
        print("Failed at create_negative_samples.")
        return None
    summary["negative_samples"] = neg

    print("Task [6/6]: Combining crash and negative samples into training data...")
    combo = combine_crash_and_negative_samples(
        crash_file=modified_file,
        negative_file="NegativeSamples.csv",
        output_filename="TrainingData.csv",
    )
    if combo is None:
        print("Failed at combine_crash_and_negative_samples.")
        return None
    summary["training_data"] = combo

    print("All tasks completed.")
    print(f"Final output: {combo.get('output_file')}")
    return summary


#   #   #   #   #   #   #   #   #   #   #   #
#               CLI                         #
#   #   #   #   #   #   #   #   #   #   #   #

if __name__ == "__main__":
    cli_source = sys.argv[1] if len(sys.argv) > 1 else "Dallas2025.csv"
    training_data_production(cli_source)