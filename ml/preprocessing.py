"""Training-data preprocessing utilities for crash risk modeling."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TARGET_COLUMN = "crash"
NUMERIC_FEATURE_COLUMNS = [
    "crashmonth",
    "crashyear",
    "crashtime_minutes",
    "hour_bucket",
    "street_crash_rate",
]
CATEGORICAL_FEATURE_COLUMNS = [
    "city",
    "county",
    "dayofweek",
    "roadclass",
    "ruralurbantype",
    "surfacecondition",
    "weathercondition",
]
TRAINING_COLUMNS = [
    "city",
    "county",
    "crashmonth",
    "crashtime",
    "crashyear",
    "dayofweek",
    "hourofday",
    "roadclass",
    "ruralurbantype",
    "streetname",
    "surfacecondition",
    "weathercondition",
    "crash",
]


def normalize_training_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned dataframe with normalized column names and null handling."""
    normalized = dataframe.copy()
    normalized.columns = [
        column.strip().lower().replace(" ", "").replace("_", "")
        for column in normalized.columns
    ]
    return normalized.fillna("Unknown")


def load_training_dataframe(
    *,
    csv_path: str,
    row_limit: int | None = None,
) -> pd.DataFrame:
    """Load and normalize training CSV input from disk."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    loaded = pd.read_csv(csv_file, nrows=row_limit)
    normalized = normalize_training_columns(loaded)

    missing_columns = [
        column for column in TRAINING_COLUMNS if column not in normalized.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Training CSV is missing expected columns: {', '.join(missing_columns)}"
        )

    return normalized[TRAINING_COLUMNS].copy()


def _parse_crash_time_minutes(value: str) -> int:
    """Convert crash-time values to minutes since midnight."""
    if not value or value == "Unknown":
        return 0

    text = str(value).strip()
    if ":" in text:
        parts = text.split(":")
        hour = int(parts[0]) if parts[0].isdigit() else 0
        minute = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return max(0, min(1439, (hour * 60) + minute))

    if text.isdigit():
        number = int(text)
        return number if number <= 1439 else ((number // 100) * 60) + (number % 100)

    return 0


def _parse_hour_bucket(value: str) -> int:
    """Extract hour number from hour bucket text."""
    if not value or value == "Unknown":
        return 0

    text = str(value).strip()
    first_hour = text.split(":")[0]
    return int(first_hour) if first_hour.isdigit() else 0


def _parse_crash_label(value: str | int | float) -> int:
    """Convert crash labels to binary integer values."""
    if isinstance(value, (int, float)):
        return 1 if int(value) > 0 else 0

    text = str(value).strip()
    first_token = text.split()[0] if text else "0"
    if first_token.isdigit():
        return 1 if int(first_token) > 0 else 0
    return 1 if text.lower() in {"true", "yes", "y"} else 0


def normalize_street_name(value: str | int | float | None) -> str:
    """Normalize street names for stable crash-rate encoding."""
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if not text:
        return "unknown"

    # Collapse punctuation and repeated whitespace to improve lookup stability.
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    token_aliases = {
        "north": "n",
        "south": "s",
        "east": "e",
        "west": "w",
        "road": "rd",
        "street": "st",
        "avenue": "ave",
        "boulevard": "blvd",
        "drive": "dr",
        "lane": "ln",
        "parkway": "pkwy",
        "highway": "hwy",
        "interstate": "ih",
        "tollway": "tl",
        "turnpike": "tpke",
        "freeway": "fwy",
    }
    normalized_tokens = [token_aliases.get(token, token) for token in text.split()]
    normalized = " ".join(normalized_tokens).strip()
    return normalized if normalized else "unknown"


def _normalize_street_name(value: str | int | float | None) -> str:
    """Backward-compatible wrapper for internal callers."""
    return normalize_street_name(value)


def build_feature_table(*, dataframe: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    """Build encoded features and binary labels from normalized training data."""
    shaped = dataframe.copy()
    shaped["crashtime_minutes"] = shaped["crashtime"].map(_parse_crash_time_minutes)
    shaped["hour_bucket"] = shaped["hourofday"].map(_parse_hour_bucket)
    shaped[TARGET_COLUMN] = shaped[TARGET_COLUMN].map(_parse_crash_label).astype(np.float32)
    shaped["streetname_norm"] = shaped["streetname"].map(normalize_street_name)

    street_crash_rate_map = (
        shaped.groupby("streetname_norm")[TARGET_COLUMN].mean().astype(np.float32).to_dict()
    )
    global_street_crash_rate = float(shaped[TARGET_COLUMN].mean())
    shaped["street_crash_rate"] = (
        shaped["streetname_norm"]
        .map(street_crash_rate_map)
        .fillna(global_street_crash_rate)
        .astype(np.float32)
    )

    numeric_features = (
        shaped[NUMERIC_FEATURE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(np.float32)
    )
    categorical_features = pd.get_dummies(
        shaped[CATEGORICAL_FEATURE_COLUMNS].astype(str),
        prefix=CATEGORICAL_FEATURE_COLUMNS,
    )
    all_features = pd.concat([numeric_features, categorical_features], axis=1).astype(
        np.float32
    )
    labels = shaped[TARGET_COLUMN]

    return {
        "features": all_features,
        "labels": labels,
        "street_crash_rate_map": street_crash_rate_map,
        "global_street_crash_rate": global_street_crash_rate,
    }


def build_inference_feature_vector(
    *,
    row: dict[str, str | int | float | None],
    feature_columns: list[str],
    street_crash_rate_map: dict[str, float] | None = None,
    global_street_crash_rate: float | None = None,
) -> np.ndarray:
    """Encode a single inference row into the trained feature-column order."""
    frame = pd.DataFrame([row])
    normalized = normalize_training_columns(frame)

    for column in TRAINING_COLUMNS:
        if column == TARGET_COLUMN:
            continue
        if column not in normalized.columns:
            normalized[column] = "Unknown"

    normalized["crashtime_minutes"] = normalized["crashtime"].map(_parse_crash_time_minutes)
    normalized["hour_bucket"] = normalized["hourofday"].map(_parse_hour_bucket)
    street_name = normalize_street_name(normalized["streetname"].iloc[0])
    default_rate = 0.0 if global_street_crash_rate is None else global_street_crash_rate
    if street_crash_rate_map is None:
        street_rate = default_rate
    else:
        street_rate = float(street_crash_rate_map.get(street_name, default_rate))
    normalized["street_crash_rate"] = np.float32(street_rate)

    numeric_features = (
        normalized[NUMERIC_FEATURE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(np.float32)
    )
    categorical_features = pd.get_dummies(
        normalized[CATEGORICAL_FEATURE_COLUMNS].astype(str),
        prefix=CATEGORICAL_FEATURE_COLUMNS,
    )
    all_features = pd.concat([numeric_features, categorical_features], axis=1).astype(
        np.float32
    )
    aligned = all_features.reindex(columns=feature_columns, fill_value=0.0)
    return aligned.iloc[0].to_numpy(dtype=np.float32)


def split_feature_table(
    *,
    features: pd.DataFrame,
    labels: pd.Series,
    test_ratio: float = 0.2,
    seed: int = 7,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Split features and labels into train/test partitions."""
    if len(features) != len(labels):
        raise ValueError("Features and labels length mismatch.")

    sample_count = len(features)
    random_state = np.random.default_rng(seed=seed)
    indices = np.arange(sample_count)
    random_state.shuffle(indices)

    test_size = int(sample_count * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    return {
        "x_train": features.iloc[train_indices].reset_index(drop=True),
        "x_test": features.iloc[test_indices].reset_index(drop=True),
        "y_train": labels.iloc[train_indices].reset_index(drop=True),
        "y_test": labels.iloc[test_indices].reset_index(drop=True),
    }


def to_tensor_dataset(
    *,
    split_data: dict[str, pd.DataFrame | pd.Series],
) -> dict[str, torch.Tensor | int]:
    """Convert split dataframe partitions to torch tensors."""
    x_train = torch.tensor(split_data["x_train"].to_numpy(), dtype=torch.float32)
    x_test = torch.tensor(split_data["x_test"].to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(split_data["y_train"].to_numpy(), dtype=torch.float32)
    y_test = torch.tensor(split_data["y_test"].to_numpy(), dtype=torch.float32)

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "input_size": x_train.shape[1],
    }
