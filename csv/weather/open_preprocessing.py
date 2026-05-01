from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path


import pandas as pd

Preprocessor = Callable[[pd.DataFrame], pd.DataFrame]

_TEMP_COL = "temperature_2m (°C)"
_PRECIP_COL = "precipitation (mm)"
_WEATHER_CODE_COL = "weather_code (wmo code)"


#   #   #   #   #   #   #   #   #   #   #   #
#           Seperate date and hour          #
#   #   #   #   #   #   #   #   #   #   #   #


def seperate_date_and_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace column ``time`` (e.g. ``2025-01-01T00:00``) with ``date`` (YYYY-MM-DD)
    and ``hour`` (integer hour from the part after ``T``, minutes ignored).
    No-op if ``time`` is not present (e.g. file already processed).
    """
    if "time" not in df.columns:
        return df
    time_str = df["time"].astype(str)
    split_t = time_str.str.split("T", n=1, expand=True)
    date_part = split_t[0]
    after_t = split_t[1]
    hour_part = after_t.str.split(":", n=1, expand=True)[0].astype(int)
    out = df.drop(columns=["time"])
    out.insert(0, "hour", hour_part)
    out.insert(0, "date", date_part)
    return out


#   #   #   #   #   #   #   #   #   #
#           Add Ice Flag            #
#   #   #   #   #   #   #   #   #   # 


def add_ice_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``ice_flag`` (0/1): 1 if temperature < 0 °C and precipitation > 0 mm;
    else 1 if the previous row's ``ice_flag`` is 1 and temperature < 5 °C;
    else 0. Rows are evaluated in file order (top to bottom).
    """
    if _TEMP_COL not in df.columns or _PRECIP_COL not in df.columns:
        return df
    t = pd.to_numeric(df[_TEMP_COL], errors="coerce")
    p = pd.to_numeric(df[_PRECIP_COL], errors="coerce").fillna(0.0)
    n = len(df)
    flags: list[int] = [0] * n
    prev = 0
    for i in range(n):
        ti, pi = t.iloc[i], float(p.iloc[i])
        if pd.notna(ti) and ti < 0 and pi > 0:
            flags[i] = 1
        elif prev == 1 and pd.notna(ti) and ti < 5:
            flags[i] = 1
        else:
            flags[i] = 0
        prev = flags[i]
    out = df.copy()
    if "ice_flag" in out.columns:
        out = out.drop(columns=["ice_flag"])
    out["ice_flag"] = flags
    return out

#   #   #   #   #   #   #   #   #   #   #   #   # 
#           Set Weather/Road Conditions         #
#   #   #   #   #   #   #   #   #   #   #   #   #

# Human-readable labels for selected WMO weather_code values (see Open-Meteo docs).
_WEATHER_CODE_LABELS: dict[int, str] = {
    **{code: "1 - DRY" for code in (0, 1)},
    **{code: "2 - CLOUDY" for code in (2, 3)},
    **{code: "3 - RAIN" for code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95)},
    **{code: "4 - SLEET/HAIL" for code in (96, 99)},
    **{code: "5 - SNOW" for code in (71, 73, 75, 77, 85, 86)},
    **{code: "6 - FOG" for code in (45, 48)},
}

# Uses the Open-Meteo weather code plus temperature to estimate the road condition.
def _get_road_code_label(weather_code: int, temperature: float | None) -> str:
    """Map the weather code and temperature to a road-condition label."""
    dry_codes = (0, 1, 2, 3, 45, 48)
    wet_codes = (51, 53, 55, 56, 57, 61, 63, 66, 80, 81, 95, 96, 99)
    standing_water_codes = (65, 67, 82)
    snow_codes = (71, 73, 75, 77, 85, 86)

    if weather_code in dry_codes:
        return "1 - DRY"

    if weather_code in snow_codes:
        return "4 - SNOW"

    if weather_code in standing_water_codes:
        return "6 - ICE" if temperature is not None and temperature <= 0 else "3 - STANDING WATER"

    if weather_code in wet_codes:
        return "6 - ICE" if temperature is not None and temperature <= 0 else "2 - WET"

    return "99 - UNKNOWN"


_WEATHER_IS_SNOW = "5 - SNOW"
_ROAD_SLUSH = "5 - SLUSH"


def _compute_road_column(
    *,
    weather_codes: pd.Series,
    temperatures: pd.Series,
    ice_flags: pd.Series,
    weather_labels: pd.Series,
) -> list[str]:
    """
    Stateful ``Road`` labels: dry / wet / standing water follow ``_get_road_code_label``;
    snow carry and (0, 5] °C melt slush as in ``set_weather_road_conditions``; ice carry
    ends on ``ice_flag`` 0 (then base label) or on T > 2.5 °C (then slush); snow weather
    wins over ice when both apply.
    """
    n = len(weather_codes)
    roads: list[str] = [""] * n
    snow_carry = False
    ice_carry = False
    snow_melt = False

    for i in range(n):
        code = int(weather_codes.iloc[i]) if pd.notna(weather_codes.iloc[i]) else -1
        t_raw = temperatures.iloc[i]
        tnum: float | None = float(t_raw) if pd.notna(t_raw) else None
        ice = int(ice_flags.iloc[i]) if pd.notna(ice_flags.iloc[i]) else 0
        w = str(weather_labels.iloc[i])
        base = _get_road_code_label(code, tnum)
        snow_now = w == _WEATHER_IS_SNOW

        if snow_now:
            snow_carry = True
            ice_carry = False
            snow_melt = False
            roads[i] = "4 - SNOW"
            continue

        if snow_carry:
            exited = (tnum is not None and tnum > 0) or ice == 0
            if not exited:
                roads[i] = "4 - SNOW"
                continue
            snow_carry = False
            ice_carry = False
            if tnum is not None and 0 < tnum <= 5:
                snow_melt = True
                roads[i] = _ROAD_SLUSH
            else:
                snow_melt = False
                roads[i] = base
            continue

        if snow_melt:
            if tnum is not None and 0 < tnum <= 5:
                ice_carry = False
                roads[i] = _ROAD_SLUSH
                continue
            snow_melt = False

        if not snow_carry and not snow_melt and not snow_now and base == "6 - ICE":
            ice_carry = True

        if ice_carry:
            if ice == 0:
                ice_carry = False
                roads[i] = base
                continue
            if tnum is not None and tnum > 2.5:
                roads[i] = _ROAD_SLUSH
                ice_carry = False
                continue
            roads[i] = "6 - ICE"
            continue

        roads[i] = base

    return roads


def set_weather_road_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where WMO ``weather_code`` is 0 or 1 and ``ice_flag`` is not set (0).
    Add ``weather`` from ``_WEATHER_CODE_LABELS`` (unchanged logic).
    Add ``Road``: dry / wet / standing water match ``_get_road_code_label``; snow on the
    road persists as ``4 - SNOW`` until T > 0 °C or ``ice_flag`` becomes 0, then
    ``5 - SLUSH`` while 0 < T <= 5 °C until that band ends; freezing-road ``6 - ICE``
    persists while ``ice_flag`` is 1 (including above 0 °C until melt), snow weather
    overrides to snow, and ice becomes ``5 - SLUSH`` only once T > 2.5 °C; if
    ``ice_flag`` becomes 0 first, road falls back to the base label for that hour.
    """
    if _WEATHER_CODE_COL not in df.columns:
        return df
    out = df.copy()
    codes = pd.to_numeric(out[_WEATHER_CODE_COL], errors="coerce")
    if "ice_flag" in out.columns:
        ice = pd.to_numeric(out["ice_flag"], errors="coerce").fillna(0).astype(int)
    else:
        ice = pd.Series(0, index=out.index, dtype=int)
    drop = codes.isin((0, 1)) & (ice == 0)
    out = out.loc[~drop].reset_index(drop=True)
    codes_kept = pd.to_numeric(out[_WEATHER_CODE_COL], errors="coerce").fillna(-1).astype(int)
    out["weather"] = [_WEATHER_CODE_LABELS.get(int(c), "99 - UNKNOWN") for c in codes_kept]
    temps = pd.to_numeric(out[_TEMP_COL], errors="coerce")
    ice_kept = pd.to_numeric(out["ice_flag"], errors="coerce").fillna(0).astype(int)
    out["Road"] = _compute_road_column(
        weather_codes=codes_kept,
        temperatures=temps,
        ice_flags=ice_kept,
        weather_labels=out["weather"],
    )
    return out


#   #   #   #   #   #   #   #   #   #   #
#           Pipeline / master       #
#   #   #   #   #   #   #   #   #   #   #


def default_open_meteo_steps() -> list[Preprocessor]:
    """Ordered preprocessing steps applied by ``preprocess_open_meteo_data``."""
    return [seperate_date_and_hour, add_ice_flag, set_weather_road_conditions]


def run_preprocess_pipeline(df: pd.DataFrame, steps: Sequence[Preprocessor]) -> pd.DataFrame:
    out = df
    for step in steps:
        out = step(out)
    return out


def preprocess_open_meteo_data(
    year: int,
    *,
    steps: Sequence[Preprocessor] | None = None,
) -> None:
    """
    For each ``*_{year}.csv`` in ``weather_csvs/`` (except ``cities.csv``), skip the
    first three lines (Open-Meteo metadata), parse the hourly table, run ``steps``
    (default: date/hour split, ``add_ice_flag``, ``set_weather_road_conditions``),
    and write back preserving the original preamble.

    Pass a custom ``steps`` sequence to insert or reorder transforms without editing
    this function.
    """
    pipeline = list(steps) if steps is not None else default_open_meteo_steps()
    weather_dir = Path(__file__).resolve().parent / "weather_csvs"
    suffix = f"_{year}.csv"
    for path in sorted(weather_dir.glob("*.csv")):
        if path.name.lower() == "cities.csv":
            continue
        if not path.name.endswith(suffix):
            continue
        with path.open(encoding="utf-8") as f:
            preamble_lines = [f.readline() for _ in range(3)]
        df = pd.read_csv(path, skiprows=3)
        df = run_preprocess_pipeline(df, pipeline)
        with path.open("w", encoding="utf-8", newline="") as f:
            for line in preamble_lines:
                f.write(line)
            df.to_csv(f, index=False)
