"""
Athena v1.0
--------------------
Utility Functions

Responsibilities
----------------
1. Filesystem helpers
2. Memory optimization helpers
3. Timing utilities
4. Generic dataframe utilities
5. Serialization helpers
"""

from __future__ import annotations

import hashlib
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Union

import joblib
import numpy as np
import pandas as pd

__all__ = [
    "ensure_directory",
    "dataframe_memory_usage",
    "optimize_numeric_dtypes",
    "duplicate_count",
    "missing_value_summary",
    "timer",
    "save_pickle",
    "load_pickle",
    "sha256_checksum",
    "safe_divide",
    "clip_probability",
]

Number = Union[int, float, np.ndarray, pd.Series]


# =============================================================================
# Filesystem
# =============================================================================

def ensure_directory(path: Path) -> Path:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path : Path

    Returns
    -------
    Path
        The same directory path.

    Raises
    ------
    OSError
        Re-raised with the offending path attached, instead of a bare
        stack trace pointing at `mkdir` internals.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Could not create directory '{path}': {exc}") from exc
    return path


# =============================================================================
# Memory Helpers
# =============================================================================

def dataframe_memory_usage(df: pd.DataFrame, deep: bool = False) -> float:
    """
    Return dataframe memory usage in MB.

    Parameters
    ----------
    df : pd.DataFrame
    deep : bool, default False
        `deep=True` walks every object-dtype column to measure the actual
        Python-object memory footprint. That's an O(n) pass over every
        string/object cell, not a lookup — cheap on a small profiling
        dataframe, expensive on a multi-million-row PaySim frame called
        repeatedly during pipeline stages. Default False (fast, sizes
        object columns by pointer width only); pass True only when the
        precise figure is actually needed (e.g. a one-off audit report).
    """
    return df.memory_usage(deep=deep).sum() / (1024 ** 2)


def optimize_numeric_dtypes(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Downcast numeric columns to reduce memory usage.

    Parameters
    ----------
    df : pd.DataFrame
    columns : Iterable[str], optional
        Restrict downcasting to a subset of columns. Defaults to all
        integer/float columns.
    inplace : bool, default False
        If True, mutate and return `df` directly instead of taking a full
        `.copy()` first. `.copy()` on a multi-million-row frame briefly
        doubles peak memory for a step whose entire purpose is *reducing*
        memory pressure — callers that already hold the only reference to
        `df` (the common case, since this is meant to run first in the
        pipeline per the established "optimize-memory-before-downstream-
        passes" pattern) should pass `inplace=True`.

    Returns
    -------
    pd.DataFrame
    """
    result = df if inplace else df.copy()

    numeric_df = result[columns] if columns is not None else result
    integer_columns = numeric_df.select_dtypes(include=["int"]).columns
    float_columns = numeric_df.select_dtypes(include=["float"]).columns

    for column in integer_columns:
        result[column] = pd.to_numeric(result[column], downcast="integer")

    for column in float_columns:
        result[column] = pd.to_numeric(result[column], downcast="float")

    return result


# =============================================================================
# DataFrame Helpers
# =============================================================================

def duplicate_count(df: pd.DataFrame, subset: Iterable[str] | None = None) -> int:
    """
    Count duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
    subset : Iterable[str], optional
        Check duplicates on a subset of columns (e.g. transaction fields
        excluding a generated ID) instead of the full row.
    """
    return int(df.duplicated(subset=subset).sum())


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return missing-value statistics, sorted by severity (most missing first).
    """
    # Compute the null mask once. The original called `df.isna()` twice
    # (once for .sum(), once inside .mean()) — two full O(rows x cols)
    # passes over the entire frame for numbers that are trivially derivable
    # from each other once you have the row count.
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df) * 100).round(2) if len(df) else missing_count * 0.0

    summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent": missing_percent,
    })

    return summary[summary["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )


# =============================================================================
# Timing
# =============================================================================

class _Timer:
    """Lightweight mutable box for a context-manager-scoped elapsed time."""
    __slots__ = ("elapsed",)

    def __init__(self) -> None:
        self.elapsed = 0.0


@contextmanager
def timer():
    """
    Simple execution timer.

    Example
    -------
    with timer() as t:
        train_model()

    print(t.elapsed)

    Notes
    -----
    `elapsed` is recorded in a `finally` block, so it's still populated
    (with the time elapsed up to the failure) even if the timed code
    raises — the original left `elapsed` at 0.0 in that case, which is
    misleading when timing something you're trying to debug.
    """
    t = _Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed = time.perf_counter() - start


# =============================================================================
# Serialization
# =============================================================================

def save_pickle(obj: Any, path: Path, compress: int | bool = 3) -> None:
    """
    Persist a Python object to disk.

    Uses joblib rather than raw `pickle`: joblib handles large NumPy
    arrays (the bulk of a fitted sklearn/XGBoost/LightGBM model's state)
    far more efficiently than stdlib pickle, and `compress` gets you
    smaller model artifacts almost for free — relevant here since
    `best_model.pkl` / `scaler.pkl` / `encoder.pkl` are exactly that kind
    of object. Falls back to pickle protocol internally, so anything
    picklable still works.

    Writes to a temp file in the same directory and atomically renames it
    into place (`os.replace`), so a crash or interrupt mid-write can never
    leave a truncated, unloadable model artifact at `path` — it either
    finishes cleanly or the previous file is untouched.

    Parameters
    ----------
    obj : Any
    path : Path
    compress : int or bool, default 3
        joblib compression level (0-9). 3 is a good size/speed tradeoff
        for model artifacts; pass 0 to disable.
    """
    ensure_directory(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        joblib.dump(obj, tmp_path, compress=compress)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_pickle(path: Path) -> Any:
    """
    Load a previously-saved object.

    Note: deserialization (joblib/pickle) executes arbitrary code embedded
    in the file. Only ever load artifacts this pipeline itself wrote —
    never a file of unknown or external origin.
    """
    if not path.exists():
        raise FileNotFoundError(f"No artifact found at '{path}'")
    return joblib.load(path)


def sha256_checksum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute a SHA-256 checksum of a file, streamed in chunks so multi-MB
    model artifacts don't need to be read fully into memory at once.

    Intended for model-provenance / audit-trail use (verifying a deployed
    `best_model.pkl` matches what training produced) — the same integrity
    concern the pipeline already applies to PII via salted SHA-256
    pseudonymization.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# =============================================================================
# Numeric Helpers
# =============================================================================

def safe_divide(
    numerator: Number,
    denominator: Number,
    epsilon: float = 1e-6,
) -> Number:
    """
    Divide, substituting `epsilon` for the denominator only where it's
    (near) zero — supports scalars, NumPy arrays, and pandas Series.

    The original always computed `numerator / (denominator + epsilon)`,
    which nudges *every* result by epsilon, not just the ones that would
    otherwise divide by zero. For a balance ratio feature like
    `new_balance / old_balance` with old_balance in the thousands, that's
    negligible — but for near-zero balances (exactly the regime fraud
    detection cares about, e.g. accounts drained to ~0) it can measurably
    bias the ratio. Only substituting where needed keeps normal divisions
    exact and still avoids the div-by-zero/inf case.
    """
    if isinstance(denominator, (pd.Series, np.ndarray)):
        safe_denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
        if isinstance(denominator, pd.Series):
            safe_denominator = pd.Series(safe_denominator, index=denominator.index)
    else:
        safe_denominator = denominator if abs(denominator) >= epsilon else epsilon

    return numerator / safe_denominator


def clip_probability(values: Number) -> Number:
    """
    Clip probabilities into [0, 1].
    """
    return np.clip(values, 0.0, 1.0)