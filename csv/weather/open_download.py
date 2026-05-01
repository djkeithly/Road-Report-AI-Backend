"""
Download Open-Meteo archive hourly weather as CSV into ``weather_csvs/``.

Paths are resolved relative to this file's directory (same pattern as ``weather_download.py``).
"""

from __future__ import annotations

import csv
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

__all__ = ["get_city_list", "open_download"]

# No need for a global output path since this module writes one file per city with a predictable name.
def _script_dir() -> Path:
	return Path(__file__).resolve().parent

# Windows file names cannot contain these characters, so replace them with underscores to avoid errors.
def _safe_name_for_file(name: str) -> str:
	"""Strip and replace characters that are invalid in file names (Windows-safe)."""
	s = name.strip()
	for c in r'<>:"/\|?*':
		s = s.replace(c, "_")
	return s or "place"

def get_city_list() -> list[tuple[str, float, float]]:
	"""
	Get a list of cities with their latitudes and longitudes from ``weather_csvs/cities.csv``.

	CSV columns: ``name``, ``lat``, ``lon`` (same as ``write_cities_csv`` in ``csv_lookup``).

	Returns:
		A list of ``(name, lat, lon)`` tuples.
	"""
	path = _script_dir() / "weather_csvs" / "cities.csv"
	if not path.is_file():
		raise FileNotFoundError(f"City list not found: {path}")

	result: list[tuple[str, float, float]] = []
	with path.open(newline="", encoding="utf-8-sig") as f:
		for row in csv.DictReader(f):
			name = (row.get("name") or "").strip()
			if not name:
				continue
			try:
				lat = float(row["lat"])
				lon = float(row["lon"])
			except (KeyError, TypeError, ValueError):
				continue
			result.append((name, lat, lon))
	return result


def open_download(year: int) -> Path:
	"""
	Fetch Open-Meteo archive data for each city in ``weather_csvs/cities.csv`` for one calendar year
	and save CSVs under ``weather_csvs/``.

	Args:
		year: Calendar year (``start_date`` = Jan 1, ``end_date`` = Dec 31).

	Returns:
		Path to the last written CSV file (one file per city, ``{name}_{year}.csv`` under ``weather_csvs/``).

	Raises:
		FileNotFoundError: If ``cities.csv`` is missing.
		urllib.error.URLError: Network or HTTP failure.
		ValueError: If the city list is empty or the response is not HTTP 200.
	"""
	start_date = f"{year:04d}-01-01"
	end_date = f"{year:04d}-12-31"

	cities = get_city_list()
	if not cities:
		raise ValueError("City list is empty, cannot download weather data")

	for name, lat, lon in cities:
		params = {
			"latitude": lat,
			"longitude": lon,
			"start_date": start_date,
			"end_date": end_date,
			"hourly": "temperature_2m,precipitation,weather_code",
			"format": "csv",
		}
		query = urllib.parse.urlencode(params)
		url = f"{OPEN_METEO_ARCHIVE}?{query}"

		out_dir = _script_dir() / "weather_csvs"
		out_dir.mkdir(parents=True, exist_ok=True)
		stem = _safe_name_for_file(name)
		dest = out_dir / f"{stem}_{year}.csv"

		req = urllib.request.Request(
			url,
			headers={"User-Agent": "RoadReportWeather/1.0"},
		)
		with urllib.request.urlopen(req, timeout=120) as resp:
			status = getattr(resp, "status", 200)
			if status != 200:
				raise ValueError(f"Open-Meteo archive request failed with HTTP status {status}")
			body = resp.read()

		if not body:
			print(f"Warning: Open-Meteo archive response for {name} ({lat}, {lon}) in {year} is empty")

		dest.write_bytes(body)
	
	return dest
