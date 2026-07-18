"""
Athena v1.2
--------------------
Risk Feature Engineering
Responsibilities
----------------
1. High-risk transaction indicators
2. Composite risk score
3. Rule-based fraud signals
4. Interaction features
Author: Athena
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest")

_FLAG_DTYPE = "int8"

# Column name(s) this module will look for from upstream feature modules,
# in priority order, mapped to the risk flag it feeds. Kept explicit here
# so drift between modules (renamed/removed upstream columns) is visible
# in one place rather than silently degrading to an always-zero flag.
_RAPID_ACTIVITY_COL = "rapid_sender_activity"          # from BehaviorFeatures
_FIRST_TXN_COL = "is_first_sender_transaction"         # from BehaviorFeatures
_BALANCE_ANOMALY_COL = "sender_balance_mismatch"       # from BalanceFeatures
_BALANCE_RATIO_COL = "remaining_sender_ratio"          # from BalanceFeatures

_HIGH_BALANCE_USAGE_THRESHOLD = 0.90
_HIGH_RISK_SCORE_THRESHOLD = 3
_LARGE_TXN_QUANTILE = 0.95


class RiskFeatureError(ValueError):
    """Raised when required columns are missing for risk feature engineering."""


class RiskFeatures:
    """
    Generates heuristic risk features.

    Parameters
    ----------
    large_txn_threshold:
        Fixed amount threshold above which a transaction is flagged as
        "large". If not provided, the threshold is estimated from the
        input frame's 95th percentile — but note this makes results
        dependent on batch composition. For production/inference use,
        always pass a threshold fitted on the training distribution
        (e.g. via `fit_threshold`) rather than relying on the per-call
        default, to avoid data leakage and score drift across batches.
    """

    def __init__(self, large_txn_threshold: Optional[float] = None):
        self.large_txn_threshold = large_txn_threshold

    def fit_threshold(self, df: pd.DataFrame, quantile: float = _LARGE_TXN_QUANTILE) -> "RiskFeatures":
        """Fit and store the large-transaction threshold from a reference (e.g. training) set."""
        self._validate(df)
        self.large_txn_threshold = float(df["amount"].quantile(quantile))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate(df)
        data = df.copy()

        threshold = self.large_txn_threshold
        if threshold is None:
            warnings.warn(
                "RiskFeatures.transform called without a fitted large_txn_threshold; "
                "falling back to this batch's 95th percentile. This makes "
                "'risk_large_transaction' dependent on batch composition and is "
                "not safe for production inference. Call fit_threshold() on a "
                "training reference set first.",
                stacklevel=2,
            )
            threshold = data["amount"].quantile(_LARGE_TXN_QUANTILE)

        amount = data["amount"].to_numpy()
        sender_old = data["oldbalanceOrg"].to_numpy()
        sender_new = data["newbalanceOrig"].to_numpy()
        receiver_old = data["oldbalanceDest"].to_numpy()

        # ==================================================
        # Core rule-based flags (vectorized, no per-column overhead)
        # ==================================================
        data["risk_large_transaction"] = (amount >= threshold).astype(_FLAG_DTYPE)

        data["risk_balance_drained"] = (
            (sender_old > 0) & (sender_new == 0)
        ).astype(_FLAG_DTYPE)

        data["risk_new_receiver"] = (receiver_old == 0).astype(_FLAG_DTYPE)

        # ==================================================
        # Flags sourced from upstream feature modules (behavior/balance).
        # Explicitly wired to the real column names those modules emit;
        # falls back to 0 with a warning if a module hasn't run yet,
        # rather than silently and permanently zeroing the signal.
        # ==================================================
        data["risk_rapid_activity"] = self._pull_flag(data, _RAPID_ACTIVITY_COL)
        data["risk_first_transaction"] = self._pull_flag(data, _FIRST_TXN_COL)
        data["risk_balance_anomaly"] = self._pull_flag(data, _BALANCE_ANOMALY_COL)

        if _BALANCE_RATIO_COL in data.columns:
            data["risk_high_balance_usage"] = (
                data[_BALANCE_RATIO_COL] >= _HIGH_BALANCE_USAGE_THRESHOLD
            ).astype(_FLAG_DTYPE)
        else:
            warnings.warn(
                f"'{_BALANCE_RATIO_COL}' not found; run BalanceFeatures before "
                f"RiskFeatures. 'risk_high_balance_usage' defaulting to 0.",
                stacklevel=2,
            )
            data["risk_high_balance_usage"] = np.zeros(len(data), dtype=_FLAG_DTYPE)

        # ==================================================
        # Composite risk score (single fused sum, no intermediate copies)
        # ==================================================
        risk_columns = [
            "risk_large_transaction",
            "risk_balance_drained",
            "risk_new_receiver",
            "risk_rapid_activity",
            "risk_first_transaction",
            "risk_balance_anomaly",
            "risk_high_balance_usage",
        ]
        data["risk_score"] = (
            data[risk_columns].to_numpy().sum(axis=1).astype(_FLAG_DTYPE)
        )

        # ==================================================
        # High-risk transaction
        # ==================================================
        data["high_risk_transaction"] = (
            data["risk_score"] >= _HIGH_RISK_SCORE_THRESHOLD
        ).astype(_FLAG_DTYPE)

        return data

    @staticmethod
    def _pull_flag(data: pd.DataFrame, col: str) -> np.ndarray:
        if col in data.columns:
            return data[col].astype(_FLAG_DTYPE).to_numpy()
        warnings.warn(
            f"'{col}' not found; run the upstream feature module that produces it "
            f"before RiskFeatures. Defaulting this risk flag to 0.",
            stacklevel=3,
        )
        return np.zeros(len(data), dtype=_FLAG_DTYPE)

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise RiskFeatureError(
                f"Missing required column(s) for risk feature engineering: {missing}"
            )
        