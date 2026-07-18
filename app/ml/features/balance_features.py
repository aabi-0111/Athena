"""
Athena v1.2
--------------------
Balance Feature Engineering
Responsibilities
----------------
1. Sender balance statistics
2. Receiver balance statistics
3. Balance ratios
4. Zero balance indicators
5. Balance consistency features (incl. PaySim balance-mismatch fraud signal)
Author: Athena
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)

_FLOAT_DTYPE = "float32"
_FLAG_DTYPE = "int8"


class BalanceFeatureError(ValueError):
    """Raised when required balance columns are missing or malformed."""


class BalanceFeatures:
    """
    Generates balance-related features for transaction fraud detection.

    All ratio/delta features are computed in a small number of vectorized
    passes and cast to float32 to reduce memory footprint on large frames.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate(df)
        data = df.copy()

        # Cache columns locally to avoid repeated DataFrame __getitem__ lookups
        amount = data["amount"].to_numpy(dtype=_FLOAT_DTYPE, copy=False)
        sender_old_raw = data["oldbalanceOrg"].to_numpy(dtype=_FLOAT_DTYPE, copy=False)
        sender_new = data["newbalanceOrig"].to_numpy(dtype=_FLOAT_DTYPE, copy=False)
        receiver_old_raw = data["oldbalanceDest"].to_numpy(dtype=_FLOAT_DTYPE, copy=False)
        receiver_new = data["newbalanceDest"].to_numpy(dtype=_FLOAT_DTYPE, copy=False)

        # Safe divisors (0 -> NaN so division yields NaN instead of inf)
        sender_old_safe = np.where(sender_old_raw == 0, np.nan, sender_old_raw)
        receiver_old_safe = np.where(receiver_old_raw == 0, np.nan, receiver_old_raw)

        # ==================================================
        # Deltas
        # ==================================================
        sender_delta = sender_old_raw - sender_new
        receiver_delta = receiver_new - receiver_old_raw
        data["sender_balance_delta"] = sender_delta
        data["receiver_balance_delta"] = receiver_delta

        # ==================================================
        # Percentage changes / ratios (single divide + nan_to_num pass each)
        # ==================================================
        data["sender_balance_pct_change"] = np.nan_to_num(
            sender_delta / sender_old_safe, nan=0.0, posinf=0.0, neginf=0.0
        )
        data["receiver_balance_pct_change"] = np.nan_to_num(
            receiver_delta / receiver_old_safe, nan=0.0, posinf=0.0, neginf=0.0
        )
        data["remaining_sender_ratio"] = np.nan_to_num(
            sender_new / sender_old_safe, nan=0.0, posinf=0.0, neginf=0.0
        )
        data["receiver_growth_ratio"] = np.nan_to_num(
            receiver_new / receiver_old_safe, nan=0.0, posinf=0.0, neginf=0.0
        )

        for col in (
            "sender_balance_delta",
            "receiver_balance_delta",
            "sender_balance_pct_change",
            "receiver_balance_pct_change",
            "remaining_sender_ratio",
            "receiver_growth_ratio",
        ):
            data[col] = data[col].astype(_FLOAT_DTYPE)

        # ==================================================
        # Zero balance flags (fused boolean computation)
        # ==================================================
        sender_zero_before = sender_old_raw == 0
        sender_zero_after = sender_new == 0
        receiver_zero_before = receiver_old_raw == 0
        receiver_zero_after = receiver_new == 0

        data["sender_zero_before"] = sender_zero_before.astype(_FLAG_DTYPE)
        data["sender_zero_after"] = sender_zero_after.astype(_FLAG_DTYPE)
        data["receiver_zero_before"] = receiver_zero_before.astype(_FLAG_DTYPE)
        data["receiver_zero_after"] = receiver_zero_after.astype(_FLAG_DTYPE)

        # ==================================================
        # Negative balance flag (sanity check)
        # ==================================================
        data["negative_balance_flag"] = (
            (sender_new < 0) | (receiver_new < 0)
        ).astype(_FLAG_DTYPE)

        # ==================================================
        # Sender balance exhausted (full drain)
        # ==================================================
        data["sender_balance_exhausted"] = (
            (sender_old_raw > 0) & sender_zero_after
        ).astype(_FLAG_DTYPE)

        # ==================================================
        # Balance consistency check (PaySim-specific fraud signal):
        # legitimate transactions should satisfy
        #   newbalanceOrig ≈ oldbalanceOrg - amount
        #   newbalanceDest ≈ oldbalanceDest + amount
        # Deviations (esp. destination-side zero-balance mismatches) are a
        # well-documented strong fraud indicator in PaySim-based literature.
        # ==================================================
        expected_sender_new = sender_old_raw - amount
        expected_receiver_new = receiver_old_raw + amount
        tol = 1e-2  # currency rounding tolerance

        data["sender_balance_mismatch"] = (
            np.abs(sender_new - expected_sender_new) > tol
        ).astype(_FLAG_DTYPE)
        data["receiver_balance_mismatch"] = (
            np.abs(receiver_new - expected_receiver_new) > tol
        ).astype(_FLAG_DTYPE)

        # Classic PaySim fraud tell: destination balances stay at zero
        # before AND after despite a nonzero amount changing hands.
        data["receiver_zero_balance_anomaly"] = (
            receiver_zero_before & receiver_zero_after & (amount > 0)
        ).astype(_FLAG_DTYPE)

        return data

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise BalanceFeatureError(
                f"Missing required column(s) for balance feature engineering: {missing}"
            )
        