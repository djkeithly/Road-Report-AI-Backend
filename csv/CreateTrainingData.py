from weather.csv_geocode import geocode
from weather.weather_download import create_weather_files
from csv_lookup import lookup
import sys

# Rare manual entry point.
# Transforms a file called isd-history.csv in the same directory into geocoded.csv, which is used by other scripts.
# state and ctry can be modified here, but the default is to geocode all US stations in Texas, which is the main area of interest for this project.
if __name__ == "__main__":
    state = "TX"
    ctry = "US"
    
    # Geocode step (from csv_geocode.py)
    if "--geocode" in sys.argv:
        print(f"Geocoding {ctry} {state}...")
        geocode(ctry=ctry, state=state)

    # Setup for download step
    cities = True
    counties = ["Dallas"]
    years = [ 2025]

    # Download step (from weather_download.py)
    print("Downloading weather files...")
    create_weather_files(ctry=ctry, state=state, counties=counties, cities=cities, years=years)

    #   #   #   #   #   #   #   #
    #   Preprocessing Block     #
    #   #   #   #   #   #   #   #
    
    rawCSV = "Dallas2025.csv"

    # Lookup step (from csv_lookup.py)
    print(f"Looking up places from {rawCSV}...")
    lookup(src_path=rawCSV, cities=cities)