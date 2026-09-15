"""
Athena v1.4
--------------------
Train/Test Split Module
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.constants import TRAIN_TEST_DIR, TARGET_COLUMN
from app.core.logger import get_logger

logger = get_logger(__name__)


class DataSplitError(ValueError):
    """Raised when the dataset fails validation or cannot be split."""


class DataPersistenceError(IOError):
    """Raised when split datasets cannot be written to disk."""


class DataSplitter:
    """Create reproducible stratified train/test partitions."""

    DEFAULT_OUTPUT_DIR = TRAIN_TEST_DIR
    DEFAULT_TARGET_COLUMN = TARGET_COLUMN
    _MIN_SAMPLES_PER_CLASS = 2

    def __init__(
        self,
        test_size: float = 0.20,
        random_state: int = 42,
        stratify: bool = True,
        target_column: str = DEFAULT_TARGET_COLUMN,
        output_dir: Optional[Path] = None,
        write_combined: bool = True,
        drop_identifier_columns: bool = True,
    ):
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1 (exclusive).")
        if not isinstance(random_state, int):
            raise TypeError("random_state must be an integer.")

        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.target_column = target_column
        self.write_combined = write_combined
        self.drop_identifier_columns = drop_identifier_columns
        self.output_dir = Path(output_dir) if output_dir else self.DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        self._validate(df)

        logger.info(
            "Starting train/test split | rows=%d test_size=%.2f stratify=%s",
            len(df), self.test_size, self.stratify,
        )

        # Keep identifiers when the caller still needs them for downstream
        # feature engineering. The default remains model-safe and preserves
        # the previous behaviour.
        drop_columns = [self.target_column]
        if self.drop_identifier_columns:
            drop_columns.extend(["nameOrig", "nameDest"])

        X = df.drop(columns=drop_columns, errors="ignore")
        y = df[self.target_column]
        stratify_col = y if self.stratify else None

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                shuffle=True,
                stratify=stratify_col,
            )
        except ValueError as exc:
            raise DataSplitError(f"train_test_split failed: {exc}") from exc

        self._log_class_balance(y_train, y_test)
        self._save(X_train, X_test, y_train, y_test)
        return X_train, X_test, y_train, y_test

    def _save(self, X_train, X_test, y_train, y_test) -> None:
        writes = {
            "X_train.csv": X_train,
            "X_test.csv": X_test,
            "y_train.csv": y_train.to_frame(),
            "y_test.csv": y_test.to_frame(),
        }
        if self.write_combined:
            writes["train.csv"] = pd.concat([X_train, y_train], axis=1)
            writes["test.csv"] = pd.concat([X_test, y_test], axis=1)

        for filename, frame in writes.items():
            self._atomic_write_csv(frame, self.output_dir / filename)

    @staticmethod
    def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
            )
            with os.fdopen(fd, "w", newline="") as f:
                frame.to_csv(f, index=False)
            os.replace(tmp_path, path)
        except OSError as exc:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise DataPersistenceError(f"Failed writing {path}: {exc}") from exc

    def _log_class_balance(self, y_train: pd.Series, y_test: pd.Series) -> None:
        logger.info(
            "Split completed | Train=%d (fraud=%.4f%%) Test=%d (fraud=%.4f%%)",
            len(y_train), y_train.mean() * 100,
            len(y_test), y_test.mean() * 100,
        )

    def _validate(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise DataSplitError("Cannot split an empty dataset.")
        if self.target_column not in df.columns:
            raise DataSplitError(f"Target column '{self.target_column}' not found.")
        if df.columns.duplicated().any():
            dupes = df.columns[df.columns.duplicated()].unique().tolist()
            raise DataSplitError(f"Duplicate columns found: {dupes}")

        target = df[self.target_column]
        if target.isna().any():
            raise DataSplitError(
                f"Target column '{self.target_column}' contains {int(target.isna().sum())} null values."
            )
        if target.nunique() < 2:
            raise DataSplitError("Target column must contain at least two classes.")

        if self.stratify:
            counts = target.value_counts()
            insufficient = counts[counts < self._MIN_SAMPLES_PER_CLASS]
            if not insufficient.empty:
                raise DataSplitError(
                    f"Stratified split requires >= {self._MIN_SAMPLES_PER_CLASS} samples per class; "
                    f"classes below threshold: {insufficient.to_dict()}"
                )
            min_class_count = counts.min()
            if min_class_count * self.test_size < 1 or min_class_count * (1 - self.test_size) < 1:
                raise DataSplitError(
                    f"test_size is incompatible with the minority class size ({min_class_count} samples)."
                )
