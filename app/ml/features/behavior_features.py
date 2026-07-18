"""
Athena v1.2
--------------------
Behavior Feature Engineering

Responsibilities
----------------
1. Sender transaction frequency
2. Receiver transaction frequency
3. Transaction velocity
4. Average transaction amount per sender
5. Average transaction amount per receiver
6. Deviation from user's normal behavior
7. First-time sender/receiver flags

Author: Athena
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("step", "amount", "nameOrig", "nameDest")

_COUNT_DTYPE = "int32"
_FLAG_DTYPE = "int8"
_FLOAT_DTYPE = "float32"

# Transactions this many steps apart or fewer are considered "rapid".
_RAPID_GAP_THRESHOLD = 1
_NO_PRIOR_TXN_SENTINEL = -1


class BehaviorFeatureError(ValueError):
    """Raised when required columns are missing for behavior feature engineering."""


class BehaviorFeatures:
    """
    Generates behavior-based features using sender/receiver transaction history.

    All cumulative statistics (counts, running averages, time gaps) are
    computed with vectorized groupby/cumsum operations rather than
    `.expanding()`, which is materially faster on large transaction volumes.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate(df)

        # Sort by step, using the original row order as a stable tiebreaker
        # so results are reproducible regardless of input ordering.
        data = df.copy()
        data["_orig_order"] = np.arange(len(data))
        data = data.sort_values(["step", "_orig_order"]).reset_index(drop=True)
        data = data.drop(columns="_orig_order")

        sender_group = data.groupby("nameOrig", sort=False)
        receiver_group = data.groupby("nameDest", sort=False)

        # ==================================================
        # Transaction counts (1-indexed, cumulative)
        # ==================================================
        data["sender_txn_count"] = (
            sender_group.cumcount().add(1).astype(_COUNT_DTYPE)
        )
        data["receiver_txn_count"] = (
            receiver_group.cumcount().add(1).astype(_COUNT_DTYPE)
        )

        # ==================================================
        # Time since previous transaction
        # ==================================================
        data["sender_time_gap"] = (
            data["step"] - sender_group["step"].shift(1)
        ).fillna(_NO_PRIOR_TXN_SENTINEL).astype(_FLOAT_DTYPE)

        data["receiver_time_gap"] = (
            data["step"] - receiver_group["step"].shift(1)
        ).fillna(_NO_PRIOR_TXN_SENTINEL).astype(_FLOAT_DTYPE)

        # ==================================================
        # Running average amount BEFORE current transaction
        # (vectorized: cumulative sum / cumulative count, shifted by one row)
        # ==================================================
        data["sender_avg_amount"] = self._prior_running_mean(
            data, sender_group, "nameOrig"
        )
        data["receiver_avg_amount"] = self._prior_running_mean(
            data, receiver_group, "nameDest"
        )

        # ==================================================
        # Deviation from normal amount
        # ==================================================
        data["sender_amount_deviation"] = (
            data["amount"] - data["sender_avg_amount"]
        ).astype(_FLOAT_DTYPE)
        data["receiver_amount_deviation"] = (
            data["amount"] - data["receiver_avg_amount"]
        ).astype(_FLOAT_DTYPE)

        # ==================================================
        # First transaction flags
        # ==================================================
        data["is_first_sender_transaction"] = (
            data["sender_txn_count"] == 1
        ).astype(_FLAG_DTYPE)
        data["is_first_receiver_transaction"] = (
            data["receiver_txn_count"] == 1
        ).astype(_FLAG_DTYPE)

        # ==================================================
        # Rapid transaction flags
        # ==================================================
        data["rapid_sender_activity"] = (
            (data["sender_time_gap"] >= 0)
            & (data["sender_time_gap"] <= _RAPID_GAP_THRESHOLD)
        ).astype(_FLAG_DTYPE)
        data["rapid_receiver_activity"] = (
            (data["receiver_time_gap"] >= 0)
            & (data["receiver_time_gap"] <= _RAPID_GAP_THRESHOLD)
        ).astype(_FLAG_DTYPE)

        return data

    @staticmethod
    def _prior_running_mean(
        data: pd.DataFrame, group: "pd.core.groupby.DataFrameGroupBy", key: str
    ) -> pd.Series:
        """
        Vectorized replacement for `.expanding().mean().shift(1)`.

        Computes, for each row, the mean of that entity's amounts strictly
        before the current transaction. Entities with no prior transaction
        fall back to their own current amount (matches original semantics).
        """
        cum_sum_prior = group["amount"].cumsum() - data["amount"]
        cum_count_prior = group.cumcount()  # 0-indexed count of prior rows

        with np.errstate(invalid="ignore", divide="ignore"):
            prior_mean = cum_sum_prior / cum_count_prior.replace(0, np.nan)

        return prior_mean.fillna(data["amount"]).astype(_FLOAT_DTYPE)

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise BehaviorFeatureError(
                f"Missing required column(s) for behavior feature engineering: {missing}"
            )
        