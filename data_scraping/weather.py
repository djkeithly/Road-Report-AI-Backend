import csv

import requests

URL = "https://api.weather.gov/points"

with open("Texas_Counties_Centroid_Map.csv", mode="r") as file:
    county_file = csv.reader(file)
    _ = next(county_file)
    for line in county_file:
        PARAMS = {"latitude": line[0], "longitude": line[1]}
        request = requests.get(url=URL, params=PARAMS)
        data = request.json()
        print(data)
