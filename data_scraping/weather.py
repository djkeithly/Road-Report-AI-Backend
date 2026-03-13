import csv
import json
import time

import requests

URL = "https://api.weather.gov/points/"
# Weather.gov API requires a User-Agent header to avoid 403 errors
HEADERS = {"User-Agent": "(myroadreportapp.com, contact@email.com)"}

all_forecasts = []

with open("Texas_Counties_Centroid_Map.csv", mode="r") as file:
    county_file = csv.reader(file)
    next(county_file)  # Skip header

    for line in county_file:
        # line[0] is latitude, line[1] is longitude
        lat_lon = f"{line[0]},{line[1]}"
        print(f"Fetching data for: {lat_lon}")

        try:
            # 1. Get the forecast endpoint for these coordinates
            location_req = requests.get(url=URL + lat_lon, headers=HEADERS)
            location_req.raise_for_status()
            location_data = location_req.json()

            # 2. Get the actual forecast data
            forecast_url = location_data["properties"]["forecast"]
            forecast_req = requests.get(url=forecast_url, headers=HEADERS)
            forecast_req.raise_for_status()
            forecast_data = forecast_req.json()

            # Append this specific county's data to our master list
            all_forecasts.append({"coordinates": lat_lon, "forecast": forecast_data})

        except Exception as e:
            print(f"Error fetching {lat_lon}: {e}")

# Save the collected data to a file (Overwrites existing file)
with open("all_county_weather.json", "w") as outfile:
    json.dump(all_forecasts, outfile, indent=4)

print(
    f"Successfully saved {len(all_forecasts)} county forecasts to all_county_weather.json"
)
