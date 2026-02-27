import json

json_data_files = {"weather": "all_county_weather.json"}

# Open the file in read mode ('r')
for name, json_file in json_data_files.items():
    with open(json_file, "r") as infile:
        # Load the file content into a Python list
        weather_array = json.load(infile)
        print(f"Loaded {len(weather_array)} {name} records.")
