import argparse
from pathlib import Path
from typing import Callable

import pandas as pd


# This function removes all weather CSV columns except the ones needed for modeling:
# Date, temperature, and precipitation metrics (DATE, TMP, AA1).
# It supports any input filename and can overwrite the source file or write to a new output file.
def keep_weather_columns(
	filename: str = "WeatherData.csv",
	output_filename: str | None = None,
	keep_columns: tuple[str, ...] = ("DATE", "TMP", "AA1"),
) -> Path:
	"""
	Keep only selected weather columns in a CSV file.

	Args:
		filename (str): Input CSV filename or path. Defaults to "WeatherData.csv".
		output_filename (str | None): Optional output CSV filename/path. If None,
			the input file is overwritten.
		keep_columns (tuple[str, ...]): Columns to keep in the resulting CSV.

	Returns:
		Path: The path where the filtered CSV was saved.
	"""
	csv_path = Path(filename)
	if not csv_path.is_absolute():
		csv_path = Path(__file__).parent / csv_path

	weather_df = pd.read_csv(csv_path)
	filtered_df = weather_df.loc[:, [column for column in keep_columns if column in weather_df.columns]]

	if output_filename:
		output_path = Path(output_filename)
		if not output_path.is_absolute():
			output_path = Path(__file__).parent / output_path
	else:
		output_path = csv_path

	filtered_df.to_csv(output_path, index=False)
	return output_path


# This function splits DATE values formatted like YYYY:MM:DDTHH:MM:SS.
# It saves the date-only part back to DATE and creates a Time column as HH:00-HH:59.
def split_date_and_create_time_range(
	filename: str = "WeatherData.csv",
	output_filename: str | None = None,
	date_column: str = "DATE",
	time_column: str = "Time",
	tmp_column: str = "TMP",
	aa1_column: str = "AA1",
) -> Path:
	"""
	Split weather datetime strings and normalize TMP/AA1 weather fields.

	Expected DATE format is: YYYY:MM:DDTHH:MM:SS
	Expected TMP format is: +0222,5 (keeps +0222, removes '+' and divides by 10)
	Expected AA1 format is: <number>,<rainAmount>,<junk>,<junk>

	Args:
		filename (str): Input CSV filename or path. Defaults to "WeatherData.csv".
		output_filename (str | None): Optional output CSV filename/path. If None,
			the input file is overwritten.
		date_column (str): Source and destination date column name.
		time_column (str): Name of the new time-range column.
		tmp_column (str): Source TMP column to parse and normalize.
		aa1_column (str): Source AA1 column to parse and normalize.
		observation_column (str): New column name for the AA1 observation number.

	Returns:
		Path: The path where the filtered CSV was saved.

	Raises:
		ValueError: If the DATE, TMP, or AA1 column does not exist in the CSV.
	"""
	
    # Resolve the input CSV path, supporting both absolute and relative paths.
	csv_path = Path(filename)
	if not csv_path.is_absolute():
		csv_path = Path(__file__).parent / csv_path

	weather_df = pd.read_csv(csv_path)

    # Validate that required columns exist before processing.
	if date_column not in weather_df.columns:
		raise ValueError(f"Column '{date_column}' was not found in {csv_path}")
	if tmp_column not in weather_df.columns:
		raise ValueError(f"Column '{tmp_column}' was not found in {csv_path}")
	if aa1_column not in weather_df.columns:
		raise ValueError(f"Column '{aa1_column}' was not found in {csv_path}")

    # Process the DATE column to split date and time components.
	datetime_text = weather_df[date_column].astype(str).str.strip()
	date_time_parts = datetime_text.str.partition("T")
	has_datetime_separator = datetime_text.str.contains("T", na=False)

	date_part = date_time_parts[0].str.strip()
	hour_part = date_time_parts[2].fillna("").str.strip().str.split(":", n=1).str[0]
	valid_hours = hour_part.str.fullmatch(r"\d{1,2}")
	hour_numeric = pd.to_numeric(hour_part.where(valid_hours), errors="coerce")
	valid_hour_range = hour_numeric.between(0, 23)
	if time_column in weather_df.columns:
		time_range = weather_df[time_column].astype(str)
	else:
		time_range = pd.Series("", index=weather_df.index, dtype="object")
	rows_to_update_time = has_datetime_separator & valid_hour_range
	formatted_hours = hour_numeric[valid_hour_range].astype(int).astype(str).str.zfill(2)
	time_range.loc[rows_to_update_time] = formatted_hours + ":00-" + formatted_hours + ":59"

	weather_df[date_column] = date_part
	weather_df[time_column] = time_range

	tmp_text = weather_df[tmp_column].astype(str).str.strip()
	has_tmp_comma_format = tmp_text.str.contains(",", na=False)
	tmp_parts = tmp_text.str.extract(r"^\s*([+-]?\d+)\s*(?:,.*)?$")
	parsed_tmp = pd.to_numeric(
		tmp_parts[0].str.replace("+", "", regex=False),
		errors="coerce",
	)
	current_tmp_values = pd.to_numeric(weather_df[tmp_column], errors="coerce")
	current_tmp_values.loc[has_tmp_comma_format] = (parsed_tmp / 10).loc[has_tmp_comma_format]
	weather_df[tmp_column] = current_tmp_values

	aa1_text = weather_df[aa1_column].astype(str).str.strip()
	aa1_parts = aa1_text.str.extract(r"^\s*([^,]*)\s*,\s*([^,]*)")
	has_aa1_comma_format = aa1_text.str.contains(",", na=False)
	rain_amount = pd.to_numeric(aa1_parts[1], errors="coerce")
	current_aa1_values = pd.to_numeric(weather_df[aa1_column], errors="coerce")
	current_aa1_values.loc[has_aa1_comma_format] = (rain_amount / 10).loc[has_aa1_comma_format]
	weather_df[aa1_column] = current_aa1_values

	if output_filename:
		output_path = Path(output_filename)
		if not output_path.is_absolute():
			output_path = Path(__file__).parent / output_path
	else:
		output_path = csv_path

	weather_df.to_csv(output_path, index=False)
	return output_path


# This function removes rows where AA1 has null, empty, or whitespace-only values.
# It supports any input filename and can overwrite the source file or write to a new output file.
def remove_rows_with_empty_aa1(
	filename: str = "WeatherData.csv",
	output_filename: str | None = None,
	aa1_column: str = "AA1",
) -> Path:
	"""
	Remove rows where AA1 is null or empty in a weather CSV file.

	Args:
		filename (str): Input CSV filename or path. Defaults to "WeatherData.csv".
		output_filename (str | None): Optional output CSV filename/path. If None,
			the input file is overwritten.
		aa1_column (str): Column name used for AA1 filtering.

	Returns:
		Path: The path where the filtered CSV was saved.

	Raises:
		ValueError: If the AA1 column does not exist in the CSV.
	"""
	csv_path = Path(filename)
	if not csv_path.is_absolute():
		csv_path = Path(__file__).parent / csv_path

	weather_df = pd.read_csv(csv_path)

	if aa1_column not in weather_df.columns:
		raise ValueError(f"Column '{aa1_column}' was not found in {csv_path}")

	filtered_df = weather_df[
		weather_df[aa1_column].notna() & weather_df[aa1_column].astype(str).str.strip().ne("")
	]

	if output_filename:
		output_path = Path(output_filename)
		if not output_path.is_absolute():
			output_path = Path(__file__).parent / output_path
	else:
		output_path = csv_path

	filtered_df.to_csv(output_path, index=False)
	return output_path


# This function creates an ice risk flag from temperature and precipitation.
# Rule 1: If TMP <= 0 and AA1 > 0, set ICE_FLAG = 1.
# Rule 2: If previous row ICE_FLAG is 1, keep ICE_FLAG = 1 until TMP >= 5.
def create_ice_flag(
	filename: str = "WeatherData.csv",
	output_filename: str | None = None,
	tmp_column: str = "TMP",
	precip_column: str = "AA1",
	ice_flag_column: str = "ICE_FLAG",
	freeze_threshold: float = 0.0,
	clear_threshold: float = 5.0,
	invalid_tmp_value: float = 999.9,
) -> Path:
	"""
	Create an ICE_FLAG column based on freezing temperature and precipitation persistence.

	Args:
		filename (str): Input CSV filename or path. Defaults to "WeatherData.csv".
		output_filename (str | None): Optional output CSV filename/path. If None,
			the input file is overwritten.
		tmp_column (str): Temperature column name.
		precip_column (str): Precipitation column name.
		ice_flag_column (str): Output flag column name.
		freeze_threshold (float): Temperature threshold to start ice conditions.
		clear_threshold (float): Temperature threshold to clear carry-forward ice flag.
		invalid_tmp_value (float): TMP sentinel value to remove before flagging.

	Returns:
		Path: The path where the filtered CSV was saved.

	Raises:
		ValueError: If TMP or AA1 columns do not exist in the CSV.
	"""
	csv_path = Path(filename)
	if not csv_path.is_absolute():
		csv_path = Path(__file__).parent / csv_path

	weather_df = pd.read_csv(csv_path)

	if tmp_column not in weather_df.columns:
		raise ValueError(f"Column '{tmp_column}' was not found in {csv_path}")
	if precip_column not in weather_df.columns:
		raise ValueError(f"Column '{precip_column}' was not found in {csv_path}")

	tmp_values = pd.to_numeric(weather_df[tmp_column], errors="coerce")
	weather_df = weather_df.loc[~tmp_values.eq(invalid_tmp_value)].reset_index(drop=True)
	tmp_values = pd.to_numeric(weather_df[tmp_column], errors="coerce")
	precip_values = pd.to_numeric(weather_df[precip_column], errors="coerce")

	ice_flags: list[int] = []
	previous_flag = 0

	for tmp_value, precip_value in zip(tmp_values, precip_values):
		if pd.notna(tmp_value) and tmp_value >= clear_threshold:
			current_flag = 0
		elif pd.notna(tmp_value) and pd.notna(precip_value) and tmp_value <= freeze_threshold and precip_value > 0:
			current_flag = 1
		elif previous_flag == 1:
			current_flag = 1
		else:
			current_flag = 0

		ice_flags.append(current_flag)
		previous_flag = current_flag

	weather_df[ice_flag_column] = ice_flags

	if output_filename:
		output_path = Path(output_filename)
		if not output_path.is_absolute():
			output_path = Path(__file__).parent / output_path
	else:
		output_path = csv_path

	weather_df.to_csv(output_path, index=False)
	return output_path


# This function removes rows where there is no ice risk and no precipitation.
# Rows are dropped when ICE_FLAG == 0 and AA1 == 0.
def remove_non_ice_zero_precip_rows(
	filename: str = "WeatherData.csv",
	output_filename: str | None = None,
	ice_flag_column: str = "ICE_FLAG",
	precip_column: str = "AA1",
) -> Path:
	"""
	Drop rows where ICE_FLAG is 0 and AA1 is 0.

	Args:
		filename (str): Input CSV filename or path. Defaults to "WeatherData.csv".
		output_filename (str | None): Optional output CSV filename/path. If None,
			the input file is overwritten.
		ice_flag_column (str): Ice flag column name.
		precip_column (str): Precipitation column name.

	Returns:
		Path: The path where the filtered CSV was saved.

	Raises:
		ValueError: If ICE_FLAG or AA1 columns do not exist in the CSV.
	"""
	csv_path = Path(filename)
	if not csv_path.is_absolute():
		csv_path = Path(__file__).parent / csv_path

	weather_df = pd.read_csv(csv_path)

	if ice_flag_column not in weather_df.columns:
		raise ValueError(f"Column '{ice_flag_column}' was not found in {csv_path}")
	if precip_column not in weather_df.columns:
		raise ValueError(f"Column '{precip_column}' was not found in {csv_path}")

	ice_flag_values = pd.to_numeric(weather_df[ice_flag_column], errors="coerce")
	precip_values = pd.to_numeric(weather_df[precip_column], errors="coerce")

	rows_to_drop = ice_flag_values.eq(0) & precip_values.eq(0)
	filtered_df = weather_df.loc[~rows_to_drop]

	if output_filename:
		output_path = Path(output_filename)
		if not output_path.is_absolute():
			output_path = Path(__file__).parent / output_path
	else:
		output_path = csv_path

	filtered_df.to_csv(output_path, index=False)
	return output_path



# This helper parses command-line arguments so the script can be used directly from terminal.
def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments for weather CSV filtering."""
	parser = argparse.ArgumentParser(
		description="Run selected weather-data cleaning steps on a CSV file."
	)
	parser.add_argument(
		"filename",
		nargs="?",
		default="WeatherData.csv",
		help="Input CSV filename or path (default: WeatherData.csv)",
	)
	parser.add_argument(
		"-o",
		"--output",
		dest="output_filename",
		default=None,
		help="Optional output CSV filename or path. If omitted, overwrites input file.",
	)
	return parser.parse_args()


# Registry for available cleaning steps.
# Add new functions here as they are created.
WeatherCleaningStep = Callable[[str, str | None], Path]
WEATHER_CLEANING_STEPS: dict[str, WeatherCleaningStep] = {
	"keep_columns": keep_weather_columns,
	"split_date_time": split_date_and_create_time_range,
	"remove_empty_aa1": remove_rows_with_empty_aa1,
	"create_ice_flag": create_ice_flag,
	"remove_non_ice_zero_precip_rows": remove_non_ice_zero_precip_rows,
}


# This function runs selected cleaning steps in order from parsed CLI args.
def run_weather_cleaning_steps(
	args: argparse.Namespace,
	steps_to_run: tuple[str, ...] | list[str] | None = None,
) -> Path:
	"""
	Run selected weather-cleaning steps using parsed CLI arguments.

	Args:
		args (argparse.Namespace): Parsed command-line arguments.
		steps_to_run (tuple[str, ...] | list[str] | None): Ordered cleaning step names.
			If None, runs all registered steps in mapping order.

	Returns:
		Path: The path to the final saved CSV file.

	Raises:
		ValueError: If no steps are selected or unknown step names are provided.
	"""
	selected_steps = tuple(steps_to_run) if steps_to_run is not None else tuple(WEATHER_CLEANING_STEPS.keys())

	if not selected_steps:
		raise ValueError("At least one cleaning step must be selected.")

	unknown_steps = [step for step in selected_steps if step not in WEATHER_CLEANING_STEPS]
	if unknown_steps:
		raise ValueError(f"Unknown cleaning step(s): {', '.join(unknown_steps)}")

	final_output: Path | None = None
	current_input = args.filename
	target_output = args.output_filename

	for step_index, step_name in enumerate(selected_steps):
		step_function = WEATHER_CLEANING_STEPS[step_name]
		if step_index == 0:
			final_output = step_function(current_input, target_output)
		else:
			final_output = step_function(str(final_output), str(final_output))

	return final_output


# Main function
if __name__ == "__main__":
	"""Entry point for command-line usage."""
	args = parse_args()

	# Pick steps by name, in order.
	# Examples:
	# ("keep_columns",)
	# ("split_date_time",)
	# ("remove_empty_aa1",)
	# ("create_ice_flag",)
	# ("remove_non_ice_zero_precip_rows",)
	# ("keep_columns", "split_date_time", "remove_empty_aa1", "create_ice_flag", "remove_non_ice_zero_precip_rows")
	STEPS_TO_RUN = (
		"keep_columns",
		"split_date_time",
		"remove_empty_aa1",
		"create_ice_flag",
		"remove_non_ice_zero_precip_rows",
	)

	output_path = run_weather_cleaning_steps(
		args,
		steps_to_run=STEPS_TO_RUN,
	)
	print(f"Saved filtered CSV to: {output_path}")