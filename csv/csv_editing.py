## This file will create negative samples for the model by creating new rows that represnent non-crash events. 
# It will use the existing data to create these samples, ensuring that they are realistic and varied.

import os
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
    Calculate the average number of crashes per road in the dataset.
    
    Args:
        filepath (str): Path to the input CSV file
        
    Returns:
        int: Average number of crashes per road rounded up to the nearest whole number
    """
    try:
        df = pd.read_csv(filepath)
        road_counts = df['Street Name'].value_counts()
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
        output_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(filepath))[0]}_modified.csv")
        
        # Save the modified CSV
        df_modified.to_csv(output_file, index=False)
        
        # Sort and display non-critical roads with their crash counts
        non_critical_with_counts = [(road, road_counts[road]) for road in non_critical_roads]
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
        output_file = os.path.join(script_dir, "RoadReference.csv")
        
        # Save the reference CSV
        road_reference.to_csv(output_file, index=False)
        
        # Show road type distribution
        type_counts = road_reference['Rural Urban Type'].value_counts()
        
        return {
            'unique_roads': len(road_reference),
            'output_file': output_file,
            'road_list': road_reference.to_dict('records')
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
#*************At present, it does not contain any way to reasonably predict weather ************************
def create_negative_samples(year, road_reference_file="RoadReference.csv", final_weather_file='FinalWeather.csv', output_filename="NegativeSamples.csv"):
    """
    Create negative samples (non-crash events) for every road, every hour, every day of the year.
    Uses random weather and surface conditions, but takes road attributes from road_reference.csv.
    
    Args:

        road_reference_file (str): Path to the road reference CSV
        output_filename (str): Name for the output file
        final_weather_file (str | None): Path to the final weather CSV for lookup. If None, random weather is used.
    Returns:
        dict: Summary statistics about the generated data
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))

        road_reference_path = road_reference_file
        if not os.path.isabs(road_reference_path):
            road_reference_path = os.path.join(script_dir, road_reference_path)

        weather_file_path = final_weather_file
        if weather_file_path and not os.path.isabs(weather_file_path):
            weather_file_path = os.path.join(script_dir, weather_file_path)

        # Read the road reference data
        road_df = pd.read_csv(road_reference_path)

        
        # Check if final weather file exists
        if(weather_file_path and os.path.exists(weather_file_path)):
            weather_lookup = _build_weather_lookup(weather_file_path)
            print(f"Weather lookup built from {weather_file_path} with {len(weather_lookup)} entries.")
        else:
            weather_lookup = {}
            print(f"No valid FinalWeather file found at {weather_file_path}. Cancelling negative sample generation.")
            return None

        # Days of week
        days_of_week = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
        
        # Generate all hours and dates for the year

        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59, 59)
        
        rows = []
        crash_id_counter = 30000000
        
        print("\n" + "=" * 70)
        print("GENERATING NEGATIVE SAMPLES")
        print("=" * 70)
        print(f"Total roads: {len(road_df)}")
        print(f"Generating records for every hour of the year...")
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
                        "Crash Year": year,
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
    that conflict with actual crash data (same road, date, and hour).
    
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
            # Skip if Street Name is missing
            if pd.isna(row['Street Name']):
                continue
            
            # Normalize hour regardless of source format (range or integer-like)
            hour = _extract_hour_bucket(row['Hour of Day'])
            
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
                continue
            
            # Normalize hour regardless of source format (range or integer-like)
            hour = _extract_hour_bucket(row['Hour of Day'])
            
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



#   #   #   #   #   #   #   #   #   #   #   #
#               Main Functions              #
#   #   #   #   #   #   #   #   #   #   #   #

if __name__ == "__main__":
    #   #   #   #   #   #   #   #   # 
    #           File Path           #
    #   #   #   #   #   #   #   #   # 
    file = _abs_in_script_dir("Dallas2025.csv")

    # This controls what commands get run
    # what_to_run[0] = add crash column to modified csv and remove crash id column
    # what_to_run[1] = combine street name and segment into single street name column
    # what_to_run[2] = identify and replace non critical roads and create modified csv
    # what_to_run[3] = create road reference csv
    # what_to_run[4] = create negative samples
    # what_to_run[5] = combine crash and negative samples into training data csv

    #   #   #   #   #   #   #   #   #   #   #   #   #
    #           All these should be TRUE            #
    #   #   #   #   #   #   #   #   #   #   #   #   #
    what_to_run = [True, True, False, False, False, False]

    # Gets the year
    file_year = int(pd.read_csv(file, nrows=1)["Crash Year"].iloc[0])

    if(what_to_run[0]):
        print("Task[1/6]: Adding crash column...")
        add_crash_column(file)
        print("Crash column added to modified CSV.\n")

    average_crashes = get_average_crashes_per_road(file)

    if(what_to_run[1]):
        print("Task[2/6]: Combining street name and segment...")
        combine_street_name_and_segment(file)
        print("Street name and segment combined.\n")

    if(what_to_run[2]):
        print("Task[3/6]: Removing non critical roads and creating modified CSV...")
        identify_and_replace_non_critical_roads(file, crash_threshold=average_crashes)
        print("Non critical roads replaced and modified CSV created.\n")

    modified_file = _abs_in_script_dir(f"{os.path.splitext(os.path.basename(file))[0]}_modified.csv")
    if(what_to_run[3]):
        print("Task[4/6]: Creating road reference CSV...")
        create_road_reference_csv(modified_file)
        print("Road reference CSV created.\n")

    if(what_to_run[4]):
        print("Task[5/6]: Creating negative samples...")
        create_negative_samples(file_year, road_reference_file="RoadReference.csv", final_weather_file='FinalWeather.csv', output_filename="NegativeSamples.csv")
        print("Negative samples created.\n")    

    if(what_to_run[5]):
        print("Task[6/6]: Combining crash and negative samples into training data...")
        combine_crash_and_negative_samples(crash_file=modified_file, negative_file="NegativeSamples.csv", output_filename="TrainingData.csv")
        print("Training data created.\n")
    
    print("All tasks completed.")
    print("Final output: TrainingData.csv")