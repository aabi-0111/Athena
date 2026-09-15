"""
Athena v1.0
--------------------
Validation Utilities

Responsibilities
----------------
1. Validate filesystem resources.
2. Validate dataset schema.
3. Validate dataframe integrity.
4. Validate model inputs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd

from core.constants import (
    REQUIRED_COLUMNS,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    TARGET_COLUMN,
    VALID_TRANSACTION_TYPES,
    EXPECTED_TARGET_VALUES,
)
from core.exceptions import (
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


# =============================================================================
# Internal Helpers
# =============================================================================
# `validate_categorical_columns`, `validate_target_column`, and
# `validate_target_vector` all reduce to the same question: "does this
# column/array contain any value outside an allowed set?" The original had
# that logic hand-copied three times, with target-vector doing it slightly
# differently (np.unique vs Series.unique). One implementation means one
# place to fix if the semantics (e.g. NaN handling) ever need to change.

def _find_invalid_values(values, allowed: frozenset) -> list:
    """
    Return the sorted, distinct values in `values` that are not in
    `allowed`. NaN is dropped rather than reported, since "missing" is a
    separate concern (see `validate_not_empty` / dedicated null checks) —
    otherwise every column with any missing data would spuriously report
    NaN as an "invalid value".
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    observed = set(series.dropna().unique())
    return sorted(observed - allowed)


# =============================================================================
# Filesystem Validation
# =============================================================================

def validate_file_exists(
    path: Path,
    allowed_extensions: Iterable[str] | None = None,
) -> Path:
    """
    Validate that a file exists, is a regular file, is readable, and
    (optionally) has an expected extension.

    Parameters
    ----------
    path : Path
    allowed_extensions : Iterable[str], optional
        e.g. {".csv", ".parquet"}. Skipped if not provided, so this stays
        reusable for both data files and model artifacts (.pkl) rather
        than being hardcoded to one file type.
    """
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


# =============================================================================
# DataFrame Validation
# =============================================================================

def validate_not_empty(df: pd.DataFrame) -> None:
    """
    Ensure dataframe is not empty.
    """
    if df.empty:
        raise EmptyDatasetError("Dataset contains no rows.")


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str] = REQUIRED_COLUMNS,
) -> None:
    """
    Validate required columns exist.
    """
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise MissingColumnError(
            f"Missing required columns: {sorted(missing)}"
        )


def validate_numeric_columns(df: pd.DataFrame) -> None:
    """
    Ensure numeric columns have numeric dtypes.
    """
    invalid = [
        column for column in NUMERIC_COLUMNS
        if column in df.columns and not pd.api.types.is_numeric_dtype(df[column])
    ]

    if invalid:
        raise InvalidDataTypeError(
            f"Expected numeric columns: {invalid}"
        )


def validate_categorical_columns(df: pd.DataFrame) -> None:
    """
    Validate categorical columns contain only known category values.
    """
    for column in CATEGORICAL_COLUMNS:
        if column not in df.columns:
            continue

        invalid = _find_invalid_values(df[column], VALID_TRANSACTION_TYPES)
        if invalid:
            raise DataValidationError(
                f"Invalid values in '{column}': {invalid}"
            )


def validate_target_column(df: pd.DataFrame) -> None:
    """
    Validate target labels.
    """
    if TARGET_COLUMN not in df.columns:
        raise MissingColumnError(f"'{TARGET_COLUMN}' column not found.")

    invalid = _find_invalid_values(df[TARGET_COLUMN], EXPECTED_TARGET_VALUES)
    if invalid:
        raise DataValidationError(
            f"Unexpected target values: {invalid}"
        )


def validate_duplicate_columns(df: pd.DataFrame) -> None:
    """
    Detect duplicate column names.
    """
    duplicates = df.columns[df.columns.duplicated()].tolist()

    if duplicates:
        raise DataValidationError(
            f"Duplicate column names found: {duplicates}"
        )


# =============================================================================
# Schema Validation
# =============================================================================

def validate_schema(df: pd.DataFrame) -> None:
    """
    Run complete schema validation.

    Order is deliberate, cheapest-first: emptiness and duplicate-column
    checks are O(columns); required-column presence is a set diff over
    column names; only once those pass do we run the O(rows) dtype and
    category-value scans. A malformed dataframe fails fast, before paying
    for a full-column scan that validate_required_columns would have
    caught anyway (e.g. no point checking `type` values are valid
    transaction types if `type` doesn't even exist).
    """
    validate_not_empty(df)
    validate_duplicate_columns(df)
    validate_required_columns(df)
    validate_numeric_columns(df)
    validate_categorical_columns(df)
    validate_target_column(df)


# =============================================================================
# Model Input Validation
# =============================================================================

def validate_feature_matrix(X: FeatureMatrix) -> None:
    """
    Validate a feature matrix is non-empty, fully numeric, and free of
    NaN/infinite values before it reaches training or inference.

    Checks `np.isfinite` (NaN *and* inf) rather than only `np.isnan`: a
    ratio feature computed upstream (e.g. balance ratios) can produce inf
    even when NaN-safe, and inf silently breaks StandardScaler and
    several sklearn estimators without raising — better to catch it here
    with a clear message than debug a downstream NaN-free-but-still-wrong
    training run.
    """
    if len(X) == 0:
        raise EmptyDatasetError("Feature matrix is empty.")

    array = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)

    if not np.issubdtype(array.dtype, np.number):
        raise InvalidDataTypeError(
            "Feature matrix must be fully numeric before model training/"
            f"inference (found dtype '{array.dtype}'); encode categorical "
            "columns first."
        )

    if not np.isfinite(array).all():
        raise DataValidationError(
            "Feature matrix contains NaN or infinite values."
        )


def validate_target_vector(y: TargetVector) -> None:
    """
    Validate target vector is non-empty and contains only expected labels.
    """
    if len(y) == 0:
        raise EmptyDatasetError("Target vector is empty.")

    invalid = _find_invalid_values(y, EXPECTED_TARGET_VALUES)
    if invalid:
        raise DataValidationError(
            f"Unexpected target labels: {invalid}"
        )