import pandas as pd
from pathlib import Path

stations = pd.read_csv(Path(__file__).resolve().parent / "isd-history.csv")

tx_stations = stations[stations["STATE"] == "TX"]

tx_stations["station_id"] = (
    tx_stations["USAF"].astype(str).str.zfill(6) +
    tx_stations["WBAN"].astype(str).str.zfill(5)
)

import requests
from pathlib import Path

BASE = "https://www.ncei.noaa.gov/data/global-hourly/access"
OUT = Path("noaa_data")
OUT.mkdir(exist_ok=True)

years = [2025]
stations = ["72258013960"]  # sample

for year in years:
    for station in stations:
        url = f"{BASE}/{year}/{station}.csv"
        dest = OUT / f"{station}_{year}.csv"
        print(f"Processing {station} {year}...")
        print(f"URL: {url}")

        if dest.exists():
            continue

        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
            print(f"Downloaded {dest.name}")
        else:
            print(f"Missing {station} {year}")