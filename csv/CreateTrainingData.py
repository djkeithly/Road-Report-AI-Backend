from weather.csv_geocode import geocode
from weather.weather_download import create_weather_files
import sys

# Rare manual entry point.
# Transforms a file called isd-history.csv in the same directory into geocoded.csv, which is used by other scripts.
# state and ctry can be modified here, but the default is to geocode all US stations in Texas, which is the main area of interest for this project.
if __name__ == "__main__":
    state = "TX"
    ctry = "US"
    
    if "--geocode" in sys.argv:
        print(f"Geocoding {ctry} {state}...")
        geocode(ctry=ctry, state=state)

    cities = True
    counties = ["Dallas"]
    years = [ 2025]

    
    print("Downloading weather files...")
    create_weather_files(ctry=ctry, state=state, counties=counties, cities=cities, years=years)