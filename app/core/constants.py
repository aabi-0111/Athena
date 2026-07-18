"""
Athena v1.0
--------------------
Project Constants

Responsibilities
----------------
1. Store immutable project-wide constants.
2. Centralize dataset schema.
3. Centralize filesystem layout (paths resolved relative to project root,
   so behavior is independent of the process's current working directory).
4. Avoid hardcoded values throughout the codebase.
"""

from pathlib import Path
from typing import Final

# =============================================================================
# Project Information
# =============================================================================

PROJECT_NAME: Final[str] = "Athena"
PROJECT_VERSION: Final[str] = "1.0"

# =============================================================================
# Randomness
# =============================================================================

RANDOM_STATE: Final[int] = 42

# =============================================================================
# Dataset Schema
# =============================================================================
# NOTE: tuples, not lists — these are read-only contracts. A list can be
# mutated in place by any importer (e.g. `constants.NUMERIC_COLUMNS.append(...)`),
# silently corrupting the schema for every other module that imported it.
# Tuples make that a TypeError instead of a debugging session.

# Atomic column names, defined once. Every grouping/ordering below composes
# these by name rather than re-typing string literals, so a rename can never
# drift between groups.
COL_STEP: Final[str] = "step"
COL_TYPE: Final[str] = "type"
COL_AMOUNT: Final[str] = "amount"
COL_NAME_ORIG: Final[str] = "nameOrig"
COL_OLD_BALANCE_ORIG: Final[str] = "oldbalanceOrg"
COL_NEW_BALANCE_ORIG: Final[str] = "newbalanceOrig"
COL_NAME_DEST: Final[str] = "nameDest"
COL_OLD_BALANCE_DEST: Final[str] = "oldbalanceDest"
COL_NEW_BALANCE_DEST: Final[str] = "newbalanceDest"

TARGET_COLUMN: Final[str] = "isFraud"
FLAGGED_FRAUD_COLUMN: Final[str] = "isFlaggedFraud"

ID_COLUMNS: Final[tuple[str, ...]] = (COL_NAME_ORIG, COL_NAME_DEST)
CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (COL_TYPE,)
NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    COL_STEP,
    COL_AMOUNT,
    COL_OLD_BALANCE_ORIG,
    COL_NEW_BALANCE_ORIG,
    COL_OLD_BALANCE_DEST,
    COL_NEW_BALANCE_DEST,
)

# Raw PaySim CSV column order, composed from the named constants above
# (not hand-duplicated as bare strings) so a rename anywhere above can't
# silently drift this schema check out of sync with reality.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    COL_STEP,
    COL_TYPE,
    COL_AMOUNT,
    COL_NAME_ORIG,
    COL_OLD_BALANCE_ORIG,
    COL_NEW_BALANCE_ORIG,
    COL_NAME_DEST,
    COL_OLD_BALANCE_DEST,
    COL_NEW_BALANCE_DEST,
    TARGET_COLUMN,
    FLAGGED_FRAUD_COLUMN,
)

# Convenience for feature-engineering/model-input assembly, so downstream
# modules don't each re-derive "numeric + categorical" themselves.
ALL_FEATURE_COLUMNS: Final[tuple[str, ...]] = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

# =============================================================================
# Transaction Types
# =============================================================================

# Ordered tuple: preserves a stable order for anything that encodes on it
# (e.g. OneHotEncoder categories, dashboard legends).
TRANSACTION_TYPES: Final[tuple[str, ...]] = (
    "PAYMENT",
    "TRANSFER",
    "CASH_OUT",
    "CASH_IN",
    "DEBIT",
)

# Frozenset mirror for O(1) membership checks. Feature engineering runs
# `if txn_type in TRANSACTION_TYPES` per-row over millions of rows;
# a 5-element tuple scan there is wasted work at that scale.
VALID_TRANSACTION_TYPES: Final[frozenset[str]] = frozenset(TRANSACTION_TYPES)

# The two transaction types PaySim fraud is actually injected into —
# used repeatedly across risk_features.py / behavior_features.py, so it
# belongs here rather than being re-typed as a magic tuple in each module.
FRAUD_PRONE_TRANSACTION_TYPES: Final[frozenset[str]] = frozenset({"TRANSFER", "CASH_OUT"})

# =============================================================================
# Missing Value Handling
# =============================================================================

MISSING_NUMERIC_FILL: Final[float] = 0.0
MISSING_CATEGORICAL_FILL: Final[str] = "UNKNOWN"

# =============================================================================
# Train/Test Split
# =============================================================================

TEST_SIZE: Final[float] = 0.20
VALIDATION_SIZE: Final[float] = 0.20

# =============================================================================
# Feature Engineering
# =============================================================================

EPSILON: Final[float] = 1e-6

HIGH_AMOUNT_QUANTILE: Final[float] = 0.99

SECONDS_PER_HOUR: Final[int] = 3600
HOURS_PER_DAY: Final[int] = 24
DAYS_PER_WEEK: Final[int] = 7

# =============================================================================
# Model Defaults
# =============================================================================

DEFAULT_RF_ESTIMATORS: Final[int] = 300
DEFAULT_RF_MAX_DEPTH: Final[int] = 20

DEFAULT_XGB_ESTIMATORS: Final[int] = 300
DEFAULT_XGB_LEARNING_RATE: Final[float] = 0.05
DEFAULT_XGB_MAX_DEPTH: Final[int] = 8

# =============================================================================
# Evaluation
# =============================================================================

POSITIVE_CLASS: Final[int] = 1
NEGATIVE_CLASS: Final[int] = 0

# Derived, not hand-typed: the only two labels a binary fraud target can
# legally take are the two class constants above.
EXPECTED_TARGET_VALUES: Final[frozenset[int]] = frozenset({POSITIVE_CLASS, NEGATIVE_CLASS})

DEFAULT_THRESHOLD: Final[float] = 0.50

# =============================================================================
# File Extensions
# =============================================================================

CSV_EXTENSION: Final[str] = ".csv"
MODEL_EXTENSION: Final[str] = ".pkl"
LOG_EXTENSION: Final[str] = ".log"

SUPPORTED_DATA_FILES: Final[frozenset[str]] = frozenset({".csv", ".parquet"})

# =============================================================================
# Filesystem Layout
# =============================================================================
# Resolved once, here, relative to this file — not the process cwd. Every
# other module imports these instead of building its own `Path("data/raw")`,
# which breaks the moment a script is invoked from a different directory
# (a common failure mode in tests, cron jobs, and containers).

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
TRAIN_TEST_DIR: Final[Path] = PROCESSED_DATA_DIR / "train_test"
REPORTS_DIR: Final[Path] = DATA_DIR / "reports"

MODEL_DIR: Final[Path] = PROJECT_ROOT / "ml" / "saved_models"
LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"

RAW_DATA_PATH: Final[Path] = RAW_DATA_DIR / f"paysim{CSV_EXTENSION}"
CLEANED_DATA_PATH: Final[Path] = PROCESSED_DATA_DIR / f"cleaned{CSV_EXTENSION}"
ENGINEERED_DATA_PATH: Final[Path] = PROCESSED_DATA_DIR / f"engineered{CSV_EXTENSION}"
DATA_PROFILE_REPORT_PATH: Final[Path] = REPORTS_DIR / "data_profile.txt"
FEATURE_SUMMARY_REPORT_PATH: Final[Path] = REPORTS_DIR / f"feature_summary{CSV_EXTENSION}"

# Saved-model artifact filenames. Centralized so train.py (writer) and
# predict.py (reader) can never drift apart on what a model file is called.
BEST_MODEL_PATH: Final[Path] = MODEL_DIR / f"best_model{MODEL_EXTENSION}"
SCALER_PATH: Final[Path] = MODEL_DIR / f"scaler{MODEL_EXTENSION}"
ENCODER_PATH: Final[Path] = MODEL_DIR / f"encoder{MODEL_EXTENSION}"

ATHENA_LOG_PATH: Final[Path] = LOG_DIR / f"athena{LOG_EXTENSION}"

# =============================================================================
# Logging
# =============================================================================

LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | "
    "%(name)s | %(filename)s:%(lineno)d | %(message)s"
)

LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

# =============================================================================
# Miscellaneous
# =============================================================================

BYTES_IN_MB: Final[int] = 1024 * 1024
