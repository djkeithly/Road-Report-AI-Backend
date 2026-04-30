import sys

from weather.csv_geocode import geocode
from weather.weather_download import create_weather_files
from csv_lookup import lookup
from weather.csv_weather import preprocess_all_weather_csvs
from weather.open_download import open_download
from weather.open_preprocessing import preprocess_open_meteo_data
from csv_editing import training_data_production
from road_geocaching import create_section_data, attach_average_coordinates

# Rare manual entry point.
# Transforms a file called isd-history.csv in the same directory into geocoded.csv, which is used by other scripts.
# state and ctry can be modified here, but the default is to geocode all US stations in Texas, which is the main area of interest for this project.
if __name__ == "__main__":
    state = "TX"
    ctry = "US"
    years = [2025]  # List of years to download and preprocess weather data for. Adjust as needed.
    cities = True
    rawCSV = "Dallas2025.csv"

    if "--road" in sys.argv:
        print("Creating road section data...")
        create_section_data(rawCSV)
        attach_average_coordinates(rawCSV)
        sys.exit(0)

    #   #   #   #   #   #   #   #   #
    #   Crash Preprocessing Block   #
    #   #   #   #   #   #   #   #   #

    print("Looking up cities from raw CSV...")
    lookup(rawCSV, cities=cities)

    print("Preprocessing weather CSVs...")
    open_download(2025)

    print("Preprocessing station list...")
    preprocess_open_meteo_data(years[0])  # Preprocess the Open-Meteo CSV for the first year (assumes same cities for all years).

    print("Creating training data...")
    training_data_production(rawCSV)