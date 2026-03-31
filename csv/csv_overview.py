## Main Functions for CSV Analysis:
## find_missing_data(filepath): Identifies rows with missing (NaN) values or 'no data' text, and prints a summary report.
## count_crashes_by_month(filepath): Counts and prints the number of crashes for each month,
## count_crashes_by_city(filepath): Counts and prints the number of crashes for each city.
## count_crashes_by_day(filepath): Counts and prints the number of crashes for each day of the week.
## crashes_by_city_and_day(filepath, output_format='csv'): Creates a cross-tabulation of crashes by city and day of week, and saves the results in both CSV and Excel formats for better readability.


import pandas as pd
from collections import defaultdict
import csv
import os
from statistics import median

# Identify and report missing data in the CSV file, including both NaN/null values and 'no data' text entries.
# Displays in console
# I used this one to ensure if any columns were likley to cause issues
def find_missing_data(filepath):
    """
    Find all rows with missing data or 'no data' values.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        dict: Dictionary containing:
            - 'missing_rows': List of row indices with NaN/null values
            - 'no_data_rows': List of row indices with 'no data' text
            - 'crash_ids': Corresponding Crash IDs for those rows
    """
    try:
        # Read CSV using pandas
        df = pd.read_csv(filepath)
        
        missing_rows = []
        no_data_rows = []
        crash_ids_missing = []
        crash_ids_no_data = []
        
        # Check for NaN/null values
        for idx, row in df.iterrows():
            if row.isnull().any():
                missing_rows.append(idx)
                crash_ids_missing.append(row.get('Crash ID', 'Unknown'))
        
        # Check for 'no data' text (case insensitive)
        for idx, row in df.iterrows():
            for value in row:
                if isinstance(value, str) and value.lower() == 'no data':
                    no_data_rows.append(idx)
                    crash_ids_no_data.append(row.get('Crash ID', 'Unknown'))
                    break
        
        results = {
            'missing_rows': missing_rows,
            'no_data_rows': no_data_rows,
            'crash_ids_missing': crash_ids_missing,
            'crash_ids_no_data': crash_ids_no_data
        }
        
        # Print results
        print("\nData Quality Report:")
        print("-" * 50)
        
        if missing_rows:
            print(f"Rows with missing (NaN) data: {len(missing_rows)}")
            print(f"Crash IDs: {crash_ids_missing[:10]}")  # Show first 10
            if len(crash_ids_missing) > 10:
                print(f"  ... and {len(crash_ids_missing) - 10} more")
        else:
            print("No rows with missing (NaN) data found.")
        
        if no_data_rows:
            print(f"\nRows with 'no data' text: {len(no_data_rows)}")
            print(f"Crash IDs: {crash_ids_no_data[:10]}")  # Show first 10
            if len(crash_ids_no_data) > 10:
                print(f"  ... and {len(crash_ids_no_data) - 10} more")
        else:
            print("\nNo rows with 'no data' text found.")
        
        print("-" * 50)
        
        return results
        
    except Exception as e:
        print(f"Error checking for missing data: {e}")
        return {'missing_rows': [], 'no_data_rows': [], 'crash_ids_missing': [], 'crash_ids_no_data': []}


# This function counts the number of crashes for each month and prints the results in a formatted way
# Simple but makes sure that there is enough data to work with
def count_crashes_by_month(filepath):
    """
    Read a CSV file and count the number of crashes for each month.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        Printed output of the number of crashes for each month, formatted as "Month X: Y crashes"
    """
    crashes_by_month = defaultdict(int)
    
    try:
        # Read CSV using pandas, skipping the metadata comment lines (first 6 rows)
        df = pd.read_csv(filepath)
        
        # Group by 'Crash Month' and count occurrences
        monthly_counts = df['Crash Month'].value_counts().sort_index()
        
        # Print the results
        month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
        }

        print("Crashes by Month (2025):")
        print("-" * 30)

        for month_num in range(1, 13):
            count = monthly_counts.get(month_num, 0)
            month_name = month_names.get(month_num, "Unknown")
            print(f"{month_name:12} (Month {month_num:2}): {count:5} crashes")
            crashes_by_month[month_num] = count

        print("Total Crashes: {:5} crashes".format(monthly_counts.sum()))
        print("-" * 30)

        return dict(crashes_by_month)
    except Exception as e:
        print(f"Error reading file with pandas: {e}")
        return {}

# This function counts the number of crashes for each city and prints the results in a formatted way
# Important to make sure that each city has enough data to be included in the analysis, 
# and to identify any cities that may have very few crashes which could be considered non critical for the purposes of training an AI model
def count_crashes_by_city(filepath):
    """
    Count the number of crashes for each city.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        Prints the number of crashes for each city, formatted as "City Name: X crashes"
        dict: Dictionary with city names as keys and crash counts as values
    """
    crashes_by_city = defaultdict(int)
    
    try:
        with open(filepath, 'r') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                city = row.get('City')
                if city:
                    crashes_by_city[city] += 1
        
        print("-" * 30)
        print("Crashes by City (2025):")
        for city, count in sorted(crashes_by_city.items()):
            print(f"{city:20}: {count:5} crashes")
        print("Total Crashes: {:5} crashes".format(sum(crashes_by_city.values())))
        print("-" * 30)

        return dict(sorted(crashes_by_city.items()))
    except Exception as e:
        print(f"Error reading file: {e}")
        return {}
    
# Gets the amount of crashes by day. Will be good to see if any day should be expected to be safer
# Will be useful for verifying data when overviewing the model's output
def count_crashes_by_day(filepath):
    """
    Count the number of crashes for each day of the week.
    
    Args:
        filepath (str): Path to the CSV file
    Returns:
        prints the number of crashes for each day of the week, formatted as "Day Name: X crashes"
        dict: Dictionary with day names as keys and crash counts as values
    """
    crashes_by_day = defaultdict(int)
    
    try:
        with open(filepath, 'r') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                day = row.get('Day of Week')
                if day:
                    crashes_by_day[day] += 1
        print("-" * 30)
        print("Crashes by Day of Week (2025):")

        for day, count in sorted(crashes_by_day.items()):
            print(f"{day:10}: {count:5} crashes")
        print("Total Crashes: {:5} crashes".format(sum(crashes_by_day.values())))
        print("-" * 30)
        
        return dict(sorted(crashes_by_day.items()))
    except Exception as e:
        print(f"Error reading file: {e}")
        return {}

# Creates a cross-tabulation of crashes by city and day of week, and saves the results in both CSV and Excel formats for better readability.
def crashes_by_city_and_day(filepath, output_format='csv'):
    """
    Create a cross-tabulation of crashes by city and day of week.
    Saves to both CSV and Excel formats for readability.
    
    Args:
        filepath (str): Path to the input CSV file
        output_format (str): Format to save ('csv', 'excel', or 'both')
        
    Returns:
        DataFrame: The cross-tabulation data
    """
    try:
        # Read CSV using pandas, skipping metadata
        df = pd.read_csv(filepath)
        
        # Create cross-tabulation of City and Day of Week
        crosstab = pd.crosstab(df['City'], df['Day of Week'], margins=True)
        
        # Get the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Save as CSV
        if output_format in ['csv', 'both']:
            csv_output = os.path.join(script_dir, "crashes_by_city_and_day.csv")
            crosstab.to_csv(csv_output)
            print(f"[OK] CSV file saved: {csv_output}")
        
        # Save as Excel (more readable with formatting)
        if output_format in ['excel', 'both']:
            try:
                excel_output = os.path.join(script_dir, "crashes_by_city_and_day.xlsx")
                crosstab.to_excel(excel_output)
                print(f"[OK] Excel file saved: {excel_output}")
            except ImportError:
                print("Note: openpyxl not installed. Install with: pip install openpyxl")
        
        return crosstab
        
    except Exception as e:
        print(f"Error creating cross-tabulation: {e}")
        return None

# This function counts the number of crashes for each road and calculates the median, then saves the results in both CSV and Excel formats for better readability.
def count_crashes_by_road(filepath, count_sections=False):
    """
    Count the number of crashes for each road and calculate the median.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        dict: Dictionary containing:
            - 'crashes_per_road': Dictionary with road names as keys and crash counts as values
            - 'median_crashes': Median number of crashes per road
            - 'total_roads': Total number of unique roads
            - 'total_crashes': Total number of crashes
    """
    try:
        # Read CSV file
        df = pd.read_csv(filepath)
        
        # Count crashes per road (you may need to adjust the column name)
        # Common column names: 'Road', 'Street Name', 'Road Name', 'Street'
        road_column = None
        for col_name in ['Road', 'Street Name', 'Road Name', 'Street', 'Location']:
            if col_name in df.columns:
                road_column = col_name
                break
        
        if road_column is None:
            print("Available columns:", df.columns.tolist())
            return None
        
        # Count crashes per road
        crashes_per_road = df[road_column].value_counts().to_dict()
        
        # Calculate median
        crash_counts = list(crashes_per_road.values())
        median_crashes = median(crash_counts)

        # Create DataFrame for export (sorted by crash count descending)
        export_df = pd.DataFrame([
            {'Road Name': road, 'Number of Crashes': count}
            for road, count in sorted(crashes_per_road.items(), key=lambda x: x[1], reverse=True)
        ])

        # Get the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Save as CSV
        csv_output = os.path.join(script_dir, "crashes_by_road.csv")
        export_df.to_csv(csv_output, index=False)
        print(f"\n[OK] CSV file saved: {csv_output}")
        
        # Save as Excel (more readable with formatting)
        try:
            excel_output = os.path.join(script_dir, "crashes_by_road.xlsx")
            export_df.to_excel(excel_output, index=False)
            print(f"[OK] Excel file saved: {excel_output}")
        except ImportError:
            print("Note: openpyxl not installed. Install with: pip install openpyxl")
        
        
        # Print results
        print("\n" + "=" * 60)
        print("CRASHES BY ROAD ANALYSIS")
        print("=" * 60)
        print(f"Total unique roads: {len(crashes_per_road)}")
        print(f"Total crashes: {sum(crash_counts)}")
        print(f"Median crashes per road: {median_crashes}")
        print(f"Average crashes per road: {sum(crash_counts) / len(crash_counts):.2f}")
        print("-" * 60)

        # Show top 10 roads with most crashes
        print("\nTop 10 Roads with Most Crashes:")
        sorted_roads = sorted(crashes_per_road.items(), key=lambda x: x[1], reverse=True)
        for i, (road, count) in enumerate(sorted_roads[:10], 1):
            print(f"{i:2}. {road:40} : {count:4} crashes")

        print("=" * 60)

        result = {
            'crashes_per_road': crashes_per_road,
            'median_crashes': median_crashes,
            'total_roads': len(crashes_per_road),
            'total_crashes': sum(crash_counts),
            'average_crashes': sum(crash_counts) / len(crash_counts)
        }

        if count_sections:
            section_column = 'Section'
            if section_column not in df.columns:
                print(f"\nWarning: '{section_column}' column not found. Skipping section breakdown.")
            else:
                # Group by road name and section, count crashes for each pair
                section_groups = (
                    df.groupby([road_column, section_column])
                    .size()
                    .reset_index(name='Number of Crashes')
                    .sort_values('Number of Crashes', ascending=False)
                )

                # Export section breakdown to CSV
                section_csv = os.path.join(script_dir, "crashes_by_road_section.csv")
                section_groups.to_csv(section_csv, index=False)
                print(f"\n[OK] Section breakdown CSV saved: {section_csv}")

                try:
                    section_excel = os.path.join(script_dir, "crashes_by_road_section.xlsx")
                    section_groups.to_excel(section_excel, index=False)
                    print(f"[OK] Section breakdown Excel saved: {section_excel}")
                except ImportError:
                    print("Note: openpyxl not installed. Install with: pip install openpyxl")

                print("\n" + "=" * 60)
                print("CRASHES BY ROAD & SECTION BREAKDOWN")
                print("=" * 60)
                print(f"Total unique road-section pairs: {len(section_groups)}")
                print("\nTop 10 Road Sections with Most Crashes:")
                for i, (_, row) in enumerate(section_groups.head(10).iterrows(), 1):
                    print(f"{i:2}. {str(row[road_column]):35} Section {str(row[section_column]):10} : {row['Number of Crashes']:4} crashes")
                print("=" * 60)

                result['crashes_by_section'] = section_groups.to_dict('records')

        return result
        
    except Exception as e:
        print(f"Error analyzing crashes by road: {e}")
        return None


def average_daily_traffic_overview(filepath):
    """
    Report on Average Daily Traffic Amount data coverage.

    Counts how many rows have a valid (non-'No Data') Average Daily Traffic Amount,
    and identifies roads where every single row is missing that value.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        dict: Dictionary containing:
            - 'total_rows': Total number of rows in the file
            - 'rows_with_adt': Number of rows with a valid ADT value
            - 'rows_missing_adt': Number of rows without a valid ADT value
            - 'roads_missing_all_adt': List of road names where no row has ADT data
    """
    try:
        df = pd.read_csv(filepath)

        adt_col = 'Average Daily Traffic Amount'
        road_col = 'Street Name'

        if adt_col not in df.columns:
            print(f"Error: '{adt_col}' column not found.")
            print("Available columns:", df.columns.tolist())
            return None

        # A row is considered "missing" if the value is NaN or the string 'No Data'
        has_adt = df[adt_col].apply(
            lambda v: not (pd.isna(v) or (isinstance(v, str) and v.strip().lower() == 'no data'))
        )

        rows_with_adt = int(has_adt.sum())
        rows_missing_adt = int((~has_adt).sum())
        total_rows = len(df)

        # Roads where every row is missing ADT
        roads_missing_all = []
        if road_col in df.columns:
            road_adt_counts = df.groupby(road_col).apply(lambda g: has_adt[g.index].sum())
            roads_missing_all = sorted(road_adt_counts[road_adt_counts == 0].index.tolist())

        print("\n" + "=" * 60)
        print("AVERAGE DAILY TRAFFIC AMOUNT OVERVIEW")
        print("=" * 60)
        print(f"Total rows               : {total_rows:6}")
        print(f"Rows WITH ADT data       : {rows_with_adt:6}  ({rows_with_adt / total_rows * 100:.1f}%)")
        print(f"Rows MISSING ADT data    : {rows_missing_adt:6}  ({rows_missing_adt / total_rows * 100:.1f}%)")
        print("-" * 60)

        if roads_missing_all:
            print(f"\nRoads with NO ADT data across all their rows ({len(roads_missing_all)} roads):")
            for road in roads_missing_all:
                row_count = int((df[road_col] == road).sum())
                print(f"  {road:40} ({row_count} crash row{'s' if row_count != 1 else ''})")
        else:
            print("\nAll roads have at least one row with ADT data.")

        print("=" * 60)

        return {
            'total_rows': total_rows,
            'rows_with_adt': rows_with_adt,
            'rows_missing_adt': rows_missing_adt,
            'roads_missing_all_adt': roads_missing_all,
        }

    except Exception as e:
        print(f"Error in average_daily_traffic_overview: {e}")
        return None


# Main function
# Functions do not need to be called in any particular order
# Possible functions to call:
## find_missing_data(filepath)        (Null, NaN, 'no data' text)
## count_crashes_by_month(filepath)
## count_crashes_by_city(filepath)
## count_crashes_by_day(filepath)
## crashes_by_city_and_day(filepath, output_format='both')
## count_crashes_by_road(filepath, count_sections)
## average_daily_traffic_overview(filepath)
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "my_list.csv")
    
    if not os.path.exists(csv_file):
        print(f"Error: File not found at {csv_file}")
        exit(1)

    average_daily_traffic_overview(csv_file)

    # count_crashes_by_road(csv_file, True)

    # day_data = count_crashes_by_day(csv_file)
    
    # Create and save city vs day of week analysis

    # crashes_by_city_and_day(csv_file, output_format='both')
    
