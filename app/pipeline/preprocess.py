"""
Athena v1.1
--------------------
Data Preprocessing Module (Optimized)

Responsibilities
----------------
1. Validate dataset schema & dtypes
2. Downcast numeric dtypes early (reduces memory footprint of every
   subsequent pass)
3. Handle missing values
4. Run vectorized sanity checks (fail fast, single pass)
5. Remove duplicate records
6. Encode categorical columns (with unknown-category handling)
7. Optionally pseudonymize PII columns (nameOrig / nameDest) for
   RBI data-protection compliance

Author: Athena Team
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import pandas as pd

from app.core.logger import get_logger
from app.core.exceptions import SchemaValidationError, DataIntegrityError

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Handles all preprocessing before feature engineering.

    Optimizations vs v1.0
    ----------------------
    - Numeric downcast happens FIRST, so every later full-column scan
      (dedup, sanity checks, missing-value scan) operates on smaller
      dtypes -> less memory bandwidth, better cache locality.
    - `memory_usage(deep=True)` (expensive for object/string columns)
      is only computed once and is opt-in via `report_memory`.
    - Sanity checks and missing-value checks each do a SINGLE combined
      vectorized pass instead of 3-4 separate `.any()` / `.sum()` scans.
    - `REQUIRED_COLUMNS` is a set -> O(1) membership checks.
    - Unknown transaction types no longer silently become NaN; they are
      detected, logged, and rejected explicitly (prevents silent data
      corruption feeding into the model / feature engine).
    - Optional PII pseudonymization (SHA-256 truncated hash) of
      nameOrig/nameDest for compliance, without breaking joinability
      (same account -> same hash, deterministic).
    - Custom exception types instead of generic ValueError, so callers
      (API layer) can catch/handle schema vs integrity failures
      differently and return correct HTTP status codes.
    """

    REQUIRED_COLUMNS = frozenset({
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    })

    NON_NEGATIVE_COLUMNS = (
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
    )

    TRANSACTION_MAPPING = {
        "PAYMENT": 0,
        "TRANSFER": 1,
        "CASH_OUT": 2,
        "DEBIT": 3,
        "CASH_IN": 4,
    }

    # Salt should come from config/secrets in production, not be hardcoded.
    # Injected at construction time so it never lives in source control.
    def __init__(self, dataframe: pd.DataFrame, pii_salt: Optional[str] = None):
        if dataframe is None or dataframe.empty:
            raise SchemaValidationError("Input dataframe is empty or None.")

        self.df = dataframe.copy()          # defensive copy: never mutate caller's data
        self._pii_salt = pii_salt or ""

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def validate_schema(self) -> None:
        """
        Ensure all required columns are present and numeric columns
        actually hold numeric data (protects against malformed/adversarial
        CSV uploads reaching the ML pipeline).
        """
        missing = self.REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            logger.error("Missing columns: %s", missing)
            raise SchemaValidationError(f"Dataset missing columns: {sorted(missing)}")

        numeric_cols = [
            "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest",
        ]
        non_numeric = [
            c for c in numeric_cols
            if not pd.api.types.is_numeric_dtype(self.df[c])
        ]
        if non_numeric:
            logger.error("Non-numeric values in expected numeric columns: %s", non_numeric)
            raise SchemaValidationError(f"Columns must be numeric: {non_numeric}")

        logger.info("Schema validation passed.")

    # ---------------------------------------------------------
    # Data Type Optimization (moved earlier: shrink footprint first)
    # ---------------------------------------------------------

    def optimize_memory(self, report_memory: bool = False) -> None:
        """
        Downcast numeric columns to reduce memory usage.
        Run this before dedup/sanity/missing passes so those operate on
        the smaller dtype footprint.

        `report_memory=False` skips the deep memory_usage() scan, which
        is O(n) over object columns (nameOrig/nameDest) and expensive on
        large datasets. Enable only for diagnostics, not on the hot path.
        """
        before = (
            self.df.memory_usage(deep=True).sum() / 1024 ** 2
            if report_memory else None
        )

        int_cols = self.df.select_dtypes(include=["int64"]).columns
        float_cols = self.df.select_dtypes(include=["float64"]).columns

        if len(int_cols):
            self.df[int_cols] = self.df[int_cols].apply(
                pd.to_numeric, downcast="integer"
            )
        if len(float_cols):
            self.df[float_cols] = self.df[float_cols].apply(
                pd.to_numeric, downcast="float"
            )

        if report_memory:
            after = self.df.memory_usage(deep=True).sum() / 1024 ** 2
            logger.info("Memory optimized: %.2f MB -> %.2f MB", before, after)
        else:
            logger.info("Numeric columns downcast.")

    # ---------------------------------------------------------
    # Missing Values
    # ---------------------------------------------------------

    def handle_missing_values(self) -> None:
        """
        Remove rows containing null values.
        Uses a cheap boolean short-circuit (`.values.any()`) before
        paying for the full per-column sum, since PaySim-style data is
        expected to be complete and this is the common case.
        """
        if not self.df.isnull().values.any():
            logger.info("No missing values detected.")
            return

        missing = int(self.df.isnull().sum().sum())
        logger.warning("Missing values detected: %d", missing)
        self.df = self.df.dropna()
        logger.info("Missing values removed.")

    # ---------------------------------------------------------
    # Sanity Checks (single combined pass, fail fast BEFORE dedup)
    # ---------------------------------------------------------

    def sanity_checks(self) -> None:
        """
        Vectorized integrity checks in a single pass over a small
        column subset, rather than three separate full-df `.any()` scans.
        Runs before `remove_duplicates` so bad/corrupted input is
        rejected before paying for the more expensive dedup pass.
        """
        subset = self.df[list(self.NON_NEGATIVE_COLUMNS)]
        negative_mask = subset.lt(0)

        if negative_mask.values.any():
            offending = negative_mask.any(axis=0)
            bad_cols = offending[offending].index.tolist()
            logger.error("Negative values detected in columns: %s", bad_cols)
            raise DataIntegrityError(f"Negative values found in: {bad_cols}")

        logger.info("Sanity checks passed.")

    # ---------------------------------------------------------
    # Duplicate Removal
    # ---------------------------------------------------------

    def remove_duplicates(self) -> None:
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = before - len(self.df)
        logger.info("Duplicate rows removed: %d", removed)

    # ---------------------------------------------------------
    # Encode Transaction Type
    # ---------------------------------------------------------

    def encode_transaction_type(self) -> None:
        """
        Convert transaction types into numerical labels using a
        categorical dtype (faster, lower memory than object mapping).

        Unlike v1.0, unmapped/unknown categories are detected explicitly
        rather than silently becoming NaN and corrupting the feature
        pipeline downstream.
        """
        unknown = set(self.df["type"].unique()) - set(self.TRANSACTION_MAPPING)
        if unknown:
            logger.error("Unknown transaction type(s) encountered: %s", unknown)
            raise DataIntegrityError(f"Unknown transaction type(s): {unknown}")

        self.df["type"] = (
            self.df["type"]
            .map(self.TRANSACTION_MAPPING)
            .astype("int8")
        )
        logger.info("Transaction type encoded.")

    # ---------------------------------------------------------
    # PII Pseudonymization (optional, off by default for backwards compat)
    # ---------------------------------------------------------

    def pseudonymize_identifiers(self) -> None:
        """
        Replace raw account identifiers with a deterministic truncated
        SHA-256 hash. Deterministic -> same account always maps to the
        same hash, preserving joinability for graph/network features,
        without persisting raw account numbers in logs, model artifacts,
        or SHAP explanation reports.
        """
        salt = self._pii_salt.encode("utf-8")

        def _hash(value: str) -> str:
            return hashlib.sha256(salt + str(value).encode("utf-8")).hexdigest()[:16]

        self.df["nameOrig"] = self.df["nameOrig"].map(_hash)
        self.df["nameDest"] = self.df["nameDest"].map(_hash)
        logger.info("Account identifiers pseudonymized.")

    # ---------------------------------------------------------
    # Dataset Summary
    # ---------------------------------------------------------

    def summary(self) -> None:
        fraud_count = int(self.df["isFraud"].sum())
        fraud_rate = float(self.df["isFraud"].mean() * 100)

        logger.info(
            "Dataset Summary | rows=%d cols=%d fraud_cases=%d fraud_rate=%.5f%%",
            len(self.df),
            len(self.df.columns),
            fraud_count,
            fraud_rate,
        )

    # ---------------------------------------------------------
    # Main Pipeline
    # ---------------------------------------------------------

    def process(self, pseudonymize_pii: bool = True, report_memory: bool = False) -> pd.DataFrame:
        """
        Execute preprocessing pipeline.

        Order is deliberately optimized:
        1. validate_schema     -> cheap, fail fast on structural issues
        2. optimize_memory     -> shrink dtypes before the expensive passes below
        3. handle_missing_values
        4. sanity_checks       -> fail fast on corrupt data BEFORE the costly dedup
        5. remove_duplicates
        6. encode_transaction_type
        7. pseudonymize_identifiers (optional, RBI compliance)
        8. summary
        """
        logger.info("Starting preprocessing pipeline...")

        self.validate_schema()
        self.optimize_memory(report_memory=report_memory)
        self.handle_missing_values()
        self.sanity_checks()
        self.remove_duplicates()
        self.encode_transaction_type()

        if pseudonymize_pii:
            self.pseudonymize_identifiers()

        self.summary()

        logger.info("Preprocessing completed successfully.")
        return self.df