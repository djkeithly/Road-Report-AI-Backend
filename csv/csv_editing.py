## This file will create negative samples for the model by creating new rows that represnent non-crash events. 
# It will use the existing data to create these samples, ensuring that they are realistic and varied.

import os
import pandas as pd
import random
from datetime import datetime, timedelta

# This file looks through the data set and identifies code that are non critical
# For the specific example of Dallas County, this is set any raods with less than 12 crashes
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
        
        # Count crashes per road
        road_counts = df['Street Name'].value_counts()
        
        # Identify roads with less than threshold crashes
        non_critical_roads = road_counts[road_counts < crash_threshold].index.tolist()
        
        # Create a copy of the dataframe
        df_modified = df.copy()
        
        # Replace non-critical roads with 'Non Critical Road'
        df_modified.loc[df_modified['Street Name'].isin(non_critical_roads), 'Street Name'] = 'Non Critical Road'
        
        # Get the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, "DallasCounty2025_Modified.csv")
        
        # Save the modified CSV
        df_modified.to_csv(output_file, index=False)
        
        # Print summary
        # print("\n" + "=" * 70)
        # print("NON-CRITICAL ROAD ANALYSIS AND REPLACEMENT")
        # print("=" * 70)
        # print(f"Crash threshold: {crash_threshold}")
        # print(f"Total unique roads: {len(road_counts)}")
        # print(f"Non-critical roads (< {crash_threshold} crashes): {len(non_critical_roads)}")
        # print(f"Critical roads (>= {crash_threshold} crashes): {len(road_counts) - len(non_critical_roads)}")
        # print("-" * 70)
        
        # print(f"\nNon-Critical Roads List ({len(non_critical_roads)} roads):")
        # print("-" * 70)
        
        # Sort and display non-critical roads with their crash counts
        non_critical_with_counts = [(road, road_counts[road]) for road in non_critical_roads]
        non_critical_with_counts.sort(key=lambda x: x[1], reverse=True)
        
        # for i, (road, count) in enumerate(non_critical_with_counts, 1):
        #     print(f"{i:4}. {road:50} : {count:3} crashes")
        
        # print("-" * 70)
        # print(f"\n[OK] Modified CSV saved: {output_file}")
        # print(f"[OK] All {len(non_critical_roads)} non-critical roads replaced with 'Non Critical Road'")
        # print("=" * 70)
        
        return {
            'non_critical_roads': non_critical_roads,
            'count_non_critical': len(non_critical_roads),
            'output_file': output_file,
            'non_critical_with_counts': non_critical_with_counts
        }
        
    except Exception as e:
        print(f"Error processing roads: {e}")
        return None
    

# This file creates a general reference CSV with information about each type of unique road in the dataset
# This allows the above function, identify_and_replace_non_critical_roads, to determine what roads are non critical
# It also allows the negative sampling function, create_negative_samples, to create realistic negative samples by using actual road attributes from the dataset.
def create_road_reference_csv(filepath):
    """
    Create a CSV file with unique road information from the modified dataset.
    This will be used as a reference for negative sampling.
    
    Args:
        filepath (str): Path to the modified CSV file
        
    Returns:
        dict: Dictionary containing:
            - 'unique_roads': Number of unique roads found
            - 'output_file': Path to the created reference file
            - 'road_list': List of road dictionaries
    """
    try:
        # Read the modified CSV file
        df = pd.read_csv(filepath)
        
        # Get unique combinations of Road Name, Rural Urban Type, County, and City
        # Group by Street Name and get the first occurrence of each group's attributes
        road_reference = df.groupby('Street Name').agg({
            'Rural Urban Type': 'first',
            'County': 'first',
            'City': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]  # Most common city for that road
        }).reset_index()
        
        # Rename columns for clarity
        road_reference.columns = ['Road Name', 'Rural Urban Type', 'County', 'City']
        
        # Sort by Road Name for easier reference
        road_reference = road_reference.sort_values('Road Name').reset_index(drop=True)
        
        # Get the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, "road_reference.csv")
        
        # Save the reference CSV
        road_reference.to_csv(output_file, index=False)
        
        # Print summary
        # print("\n" + "=" * 70)
        # print("ROAD REFERENCE CSV CREATION")
        # print("=" * 70)
        # print(f"Total unique roads: {len(road_reference)}")
        # print(f"Output file: {output_file}")
        # print("-" * 70)
        
        # # Show sample of the data
        # print("\nSample Road Data (first 10 roads):")
        # print("-" * 70)
        # for idx, row in road_reference.head(10).iterrows():
        #     print(f"{row['Road Name']:40} | {row['City']:20} | {row['Rural Urban Type']}")
        
        # print("-" * 70)
        
        # Show road type distribution
        # print("\nRoad Distribution by Urban/Rural Type:")
        # print("-" * 70)
        type_counts = road_reference['Rural Urban Type'].value_counts()
        for road_type, count in type_counts.items():
            print(f"{road_type:40} : {count:4} roads")
        
        # print("-" * 70)
        # print(f"\n[OK] Road reference CSV created: {output_file}")
        # print("=" * 70)
        
        return {
            'unique_roads': len(road_reference),
            'output_file': output_file,
            'road_list': road_reference.to_dict('records')
        }
        
    except Exception as e:
        print(f"Error creating road reference CSV: {e}")
        return None


# A basic function that takes a CSV file and adds a row called Crash and sets all the values in the file to 1, indicating a crash
# This allows for negative sampling, or the ability to differentiate between crash and non crash events in the dataset
# This will be critical to actually training an AI model
def add_crash_column(filepath, output_filename=None):
    """
    Add a 'Crash' column to the CSV file with all values set to 1.
    This indicates that all rows in the dataset represent crash events.
    
    Args:
        filepath (str): Path to the input CSV file
        output_filename (str): Optional custom output filename
        
    Returns:
        dict: Dictionary containing:
            - 'total_rows': Number of rows processed
            - 'output_file': Path to the new CSV file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Add the 'Crash' column with all values set to 1
        df['Crash'] = 1
        
        # Get the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Determine output filename
        if output_filename is None:
            base_name = os.path.basename(filepath)
            name_without_ext = os.path.splitext(base_name)[0]
            output_filename = f"{name_without_ext}_with_crash_label.csv"
        
        output_file = os.path.join(script_dir, output_filename)
        # Save the modified CSV
        df.to_csv(output_file, index=False)
        
        # Print summary
        # print("\n" + "=" * 70)
        # print("CRASH COLUMN ADDITION")
        # print("=" * 70)
        # print(f"Input file: {filepath}")
        # print(f"Total rows processed: {len(df)}")
        # print(f"New column added: 'Crash' (all values = 1)")
        # print("-" * 70)
        # print(f"\n[OK] Modified CSV saved: {output_file}")
        # print("=" * 70)
        
        return {
            'total_rows': len(df),
            'output_file': output_file
        }
    
    except Exception as e:
        print(f"Error adding 'Crash' column: {e}")
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

# This will loop through all the modified roads (Non Critical Road) and create negative samples for each road, each hour, and each day of the year
# This function will take a few minutes to run and will update the user of its progress in the console log
# Out put is negative_samples.csv
#*************At present, it does not contain any way to reasonably predict weather ************************
def create_negative_samples(road_reference_file, output_filename="negative_samples.csv", final_weather_file='FinalWeather.csv'):
    """
    Create negative samples (non-crash events) for every road, every hour, every day of 2025.
    Uses random weather and surface conditions, but takes road attributes from road_reference.csv.
    
    Args:
        road_reference_file (str): Path to the road reference CSV
        output_filename (str): Name for the output file
        final_weather_file (str | None): Path to the final weather CSV for lookup. If None, random weather is used.
    Returns:
        dict: Summary statistics about the generated data
    """
    try:
        # Read the road reference data
        road_df = pd.read_csv(road_reference_file)
        
        # Check if final weather file exists
        if(final_weather_file and os.path.exists(final_weather_file)):
            weather_lookup = _build_weather_lookup(final_weather_file)
            print(f"Weather lookup built from {final_weather_file} with {len(weather_lookup)} entries.")
        else:
            weather_lookup = {}
            print(f"No valid FinalWeather file found at {final_weather_file}. Cancelling negative sample generation.")
            return None

        # Days of week
        days_of_week = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
        
        # Generate all hours and dates for 2025
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 12, 31, 23, 59, 59)
        
        rows = []
        crash_id_counter = 30000000
        
        print("\n" + "=" * 70)
        print("GENERATING NEGATIVE SAMPLES")
        print("=" * 70)
        print(f"Total roads: {len(road_df)}")
        print(f"Generating records for every hour of 2025...")
        print("-" * 70)
        
        # Iterate through each road
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            day_name = current_date.strftime("%A").upper()
            if(current_date.day == 1 and current_date.hour == 0):
                print(f"Processing Month: {current_date.strftime('%B %Y')} ({date_str}) - Day: {day_name}")

            for hour in range(24):
                hour_key = f"{hour:02d}"
                hour_range = f"{hour:02d}:00 - {hour:02d}:59"

                weather_condition, surface_condition = weather_lookup.get(
                    (date_str, hour_key),
                    ("1 - CLEAR", "1 - DRY")
                )

                for _, road in road_df.iterrows():
                    rows.append({
                        "Crash ID": crash_id_counter,
                        "City": road["City"],
                        "County": road["County"],
                        "Crash Date": date_str,
                        "Crash Month": current_date.month,
                        "Crash Time": f"{hour:02d}00",
                        "Crash Year": 2025,
                        "Day of Week": day_name,
                        "Hour of Day": hour_range,
                        "Road Class": "CITY STREET",
                        "Rural Urban Type": road["Rural Urban Type"],
                        "Street Name": road["Road Name"],
                        "Surface Condition": surface_condition,
                        "Weather Condition": weather_condition,
                        "Crash": 0
                    })
                    crash_id_counter += 1

                
                # Move to next day
            current_date += timedelta(days=1)

        negative_df = pd.DataFrame(rows)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, output_filename)
        negative_df.to_csv(output_path, index=False)

        print(f"[OK] Negative samples saved: {output_path}")
        print(f"[OK] Total records: {len(negative_df):,}")

        return {"total_records": len(negative_df), "output_file": output_path}

    except Exception as e:
        print(f"Error creating negative samples: {e}")
        return None


# This function takes the samples in negative_samples.csv and combines them with the crash samples in the modified csv with the crash row
# This will runs under the assumption that one car passes on any given road at least once an hour
# IF there is a crash on a given road at a given hour, the non crash sample for that road and hour will be removed to avoid conflicting data
# This should provide a reasonable scope of crashes and non crashes
def combine_crash_and_negative_samples(crash_file, negative_file, output_filename="TrainingData.csv"):
    """
    Combine positive crash samples with negative samples, removing any negative samples
    that conflict with actual crash data (same road, date, and hour).
    
    Args:
        crash_file (str): Path to the crash labeled CSV file
        negative_file (str): Path to the negative samples CSV file
        output_filename (str): Name for the output training data file
        
    Returns:
        dict: Summary statistics about the combined dataset
    """
    try:
        print("\n" + "=" * 70)
        print("COMBINING CRASH AND NEGATIVE SAMPLES")
        print("=" * 70)
        
        # Read both CSV files
        print("Reading crash data...")
        crash_df = pd.read_csv(crash_file)
        
        print("Reading negative samples...")
        negative_df = pd.read_csv(negative_file)
        
        # Create a set of crash events (road, date, hour) for fast lookup
        crash_events = set()
        
        print("Building crash event index...")
        for _, row in crash_df.iterrows():
            # Skip if Street Name is missing
            if pd.isna(row['Street Name']):
                continue
                
            # Extract hour from 'Hour of Day' field (e.g., "23:00 - 23:59" -> "23")
            hour = row['Hour of Day'].split(':')[0]
            
            # Create a tuple key: (street_name, crash_date, hour)
            event_key = (
                str(row['Street Name']).strip().upper(),
                row['Crash Date'],
                hour
            )
            crash_events.add(event_key)
        
        print(f"Total crash events indexed: {len(crash_events)}")
        print("-" * 70)
        
        # Filter negative samples
        print("Filtering negative samples...")
        filtered_negative = []
        skipped_count = 0
        
        for _, row in negative_df.iterrows():
            # Skip if Street Name is missing
            if pd.isna(row['Street Name']):
                filtered_negative.append(row)
                continue
            
            # Extract hour from 'Hour of Day' field
            hour = row['Hour of Day'].split(':')[0]
            
            # Create event key for this negative sample
            event_key = (
                str(row['Street Name']).strip().upper(),
                row['Crash Date'],
                hour
            )
            
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

# ...existing code...
import argparse
import sys
import os

def _abs_in_script_dir(filename: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)

def run_all_pipeline(input_csv: str, weather_csv: str | None, threshold: int) -> int:
    """
    Full pipeline:
    1) identify_and_replace_non_critical_roads
    2) add_crash_column
    3) create_road_reference_csv
    4) create_negative_samples
    5) combine_crash_and_negative_samples
    """
    print("[INFO] Step 1/5: Identify and replace non-critical roads")
    step1 = identify_and_replace_non_critical_roads(input_csv, crash_threshold=threshold)
    if not step1 or "output_file" not in step1:
        print("[ERROR] Step 1 failed.")
        return 1
    modified_csv = step1["output_file"]

    print("[INFO] Step 2/5: Add crash label column")
    labeled_name = f"{os.path.splitext(os.path.basename(modified_csv))[0]}_with_crash_label.csv"
    step2 = add_crash_column(modified_csv, labeled_name)
    if not step2 or "output_file" not in step2:
        print("[ERROR] Step 2 failed.")
        return 1
    crash_labeled_csv = step2["output_file"]

    print("[INFO] Step 3/5: Create road reference CSV")
    step3 = create_road_reference_csv(modified_csv)
    if not step3 or "output_file" not in step3:
        print("[ERROR] Step 3 failed.")
        return 1
    road_reference_csv = step3["output_file"]

    print("[INFO] Step 4/5: Create negative samples")
    step4 = create_negative_samples(
        road_reference_file=road_reference_csv,
        output_filename="negative_samples.csv",
        final_weather_file=weather_csv
    )
    if not step4 or "output_file" not in step4:
        print("[ERROR] Step 4 failed.")
        return 1
    negative_csv = step4["output_file"]

    print("[INFO] Step 5/5: Combine into TrainingData.csv")
    step5 = combine_crash_and_negative_samples(
        crash_file=crash_labeled_csv,
        negative_file=negative_csv,
        output_filename="TrainingData.csv"
    )
    if not step5 or "output_file" not in step5:
        print("[ERROR] Step 5 failed.")
        return 1

    print(f"[OK] Pipeline complete: {step5['output_file']}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV editing pipeline for Road Report AI")
    parser.add_argument(
        "--mode",
        choices=["all", "replace", "label", "roadref", "negative", "combine"],
        default="all",
        help="Which step to run (default: all)"
    )
    parser.add_argument("--input", default="DallasCounty2025.csv", help="Input crash CSV")
    parser.add_argument("--modified", default="DallasCounty2025_Modified.csv", help="Modified CSV path")
    parser.add_argument("--labeled", default="DallasCounty2025_Modified_with_crash_label.csv", help="Crash-labeled CSV path")
    parser.add_argument("--roadref", default="road_reference.csv", help="Road reference CSV path")
    parser.add_argument("--negative", default="negative_samples.csv", help="Negative samples CSV path")
    parser.add_argument("--weather", default="FinalWeather.csv", help="FinalWeather CSV path")
    parser.add_argument("--threshold", type=int, default=12, help="Non-critical crash threshold")

    args = parser.parse_args()

    input_csv = _abs_in_script_dir(args.input)
    modified_csv = _abs_in_script_dir(args.modified)
    labeled_csv = _abs_in_script_dir(args.labeled)
    roadref_csv = _abs_in_script_dir(args.roadref)
    negative_csv = _abs_in_script_dir(args.negative)
    weather_csv = _abs_in_script_dir(args.weather)

    if args.mode == "all":
        sys.exit(run_all_pipeline(input_csv, weather_csv, args.threshold))

    if args.mode == "replace":
        result = identify_and_replace_non_critical_roads(input_csv, crash_threshold=args.threshold)
        sys.exit(0 if result else 1)

    if args.mode == "label":
        result = add_crash_column(modified_csv, os.path.basename(labeled_csv))
        sys.exit(0 if result else 1)

    if args.mode == "roadref":
        result = create_road_reference_csv(modified_csv)
        sys.exit(0 if result else 1)

    if args.mode == "negative":
        result = create_negative_samples(
            road_reference_file=roadref_csv,
            output_filename=os.path.basename(negative_csv),
            final_weather_file=weather_csv
        )
        sys.exit(0 if result else 1)

    if args.mode == "combine":
        result = combine_crash_and_negative_samples(
            crash_file=labeled_csv,
            negative_file=negative_csv,
            output_filename="TrainingData.csv"
        )
        sys.exit(0 if result else 1)