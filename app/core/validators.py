"""
Athena v1.0
--------------------
Validation Utilities
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd

from app.core.constants import (
    REQUIRED_COLUMNS,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    TARGET_COLUMN,
    VALID_TRANSACTION_TYPES,
    EXPECTED_TARGET_VALUES,
)
from app.core.exceptions import (
    DataValidationError,
    MissingColumnError,
    InvalidDataTypeError,
    EmptyDatasetError,
)

__all__ = [
    "validate_file_exists",
    "validate_not_empty",
    "validate_required_columns",
    "validate_numeric_columns",
    "validate_categorical_columns",
    "validate_target_column",
    "validate_duplicate_columns",
    "validate_schema",
    "validate_feature_matrix",
    "validate_target_vector",
]

FeatureMatrix = Union[np.ndarray, pd.DataFrame]
TargetVector = Union[np.ndarray, pd.Series]


def _find_invalid_values(values, allowed: frozenset) -> list:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    observed = set(series.dropna().unique())
    return sorted(observed - allowed)


def validate_file_exists(path: Path, allowed_extensions: Iterable[str] | None = None) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: '{path}'")
    if not path.is_file():
        raise FileNotFoundError(f"Expected a file but found: '{path}'")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"File exists but is not readable: '{path}'")
    if allowed_extensions is not None and path.suffix not in allowed_extensions:
        raise InvalidDataTypeError(
            f"Unsupported file extension '{path.suffix}' for '{path}'. "
            f"Expected one of {sorted(allowed_extensions)}."
        )
    return path


def validate_not_empty(df: pd.DataFrame) -> None:
    if df.empty:
        raise EmptyDatasetError("Dataset contains no rows.")


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str] = REQUIRED_COLUMNS) -> None:
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise MissingColumnError(f"Missing required columns: {sorted(missing)}")


def validate_numeric_columns(df: pd.DataFrame) -> None:
    invalid = [
        column for column in NUMERIC_COLUMNS
        if column in df.columns and not pd.api.types.is_numeric_dtype(df[column])
    ]
    if invalid:
        raise InvalidDataTypeError(f"Expected numeric columns: {invalid}")


def validate_categorical_columns(df: pd.DataFrame) -> None:
    for column in CATEGORICAL_COLUMNS:
        if column not in df.columns:
            continue
        invalid = _find_invalid_values(df[column], VALID_TRANSACTION_TYPES)
        if invalid:
            raise DataValidationError(f"Invalid values in '{column}': {invalid}")


def validate_target_column(df: pd.DataFrame) -> None:
    if TARGET_COLUMN not in df.columns:
        raise MissingColumnError(f"'{TARGET_COLUMN}' column not found.")
    invalid = _find_invalid_values(df[TARGET_COLUMN], EXPECTED_TARGET_VALUES)
    if invalid:
        raise DataValidationError(f"Unexpected target values: {invalid}")


def validate_duplicate_columns(df: pd.DataFrame) -> None:
    duplicates = df.columns[df.columns.duplicated()].tolist()
    if duplicates:
        raise DataValidationError(f"Duplicate column names found: {duplicates}")


def validate_schema(df: pd.DataFrame) -> None:
    validate_not_empty(df)
    validate_duplicate_columns(df)
    validate_required_columns(df)
    validate_numeric_columns(df)
    validate_categorical_columns(df)
    validate_target_column(df)


def validate_feature_matrix(X: FeatureMatrix) -> None:
    if len(X) == 0:
        raise EmptyDatasetError("Feature matrix is empty.")
    array = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    if not np.issubdtype(array.dtype, np.number):
        raise InvalidDataTypeError(
            "Feature matrix must be fully numeric before model training/inference "
            f"(found dtype '{array.dtype}'); encode categorical columns first."
        )
    if not np.isfinite(array).all():
        raise DataValidationError("Feature matrix contains NaN or infinite values.")


def validate_target_vector(y: TargetVector) -> None:
    if len(y) == 0:
        raise EmptyDatasetError("Target vector is empty.")
    invalid = _find_invalid_values(y, EXPECTED_TARGET_VALUES)
    if invalid:
        raise DataValidationError(f"Unexpected target labels: {invalid}")
