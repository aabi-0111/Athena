"""
Athena v1.1
--------------------
Amount Feature Engineering

Responsibilities
----------------
1. Transaction amount transformations
2. Log-scaled amount
3. Amount buckets
4. High-value transaction flag
5. Relative balance changes
6. Balance consistency checks

Author: Athena
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class AmountFeatures:
    """
    Creates amount-based features.

    All operations are vectorized for maximum speed.
    """

    def __init__(
    self,
    high_value_percentile: float = 0.95,
    high_value_threshold: float | None = None,
):
       self.high_value_percentile = high_value_percentile
       self.high_value_threshold = high_value_threshold

    def fit(self, df: pd.DataFrame) -> "AmountFeatures":
      """
      Learn the high-value transaction threshold
     from the training dataset.
    """
      self.high_value_threshold = float(
        df["amount"].quantile(self.high_value_percentile)
    )
      return self
    

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate amount-related features.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """

        data = df.copy()

        # --------------------------------------------------
        # Log Amount
        # --------------------------------------------------

        data["amount_log"] = np.log1p(data["amount"])

        # --------------------------------------------------
        # Amount Bucket
        # --------------------------------------------------

        bins = [
            -1,
            100,
            1000,
            10000,
            50000,
            np.inf,
        ]

        labels = [
            0,
            1,
            2,
            3,
            4,
        ]

        data["amount_bucket"] = pd.cut(
            data["amount"],
            bins=bins,
            labels=labels,
        ).astype("int8")

        # --------------------------------------------------
        # High Value Transaction
        # --------------------------------------------------

        if self.high_value_threshold is None:
            raise ValueError(
                "AmountFeatures has not been fitted. "
                "Call fit() before transform()."
    )

        threshold = self.high_value_threshold
    
        data["is_high_value"] = (
           data["amount"] >= threshold
        ).astype("int8")

        # --------------------------------------------------
        # Sender Balance Used Ratio
        # --------------------------------------------------

        sender_balance = data["oldbalanceOrg"].replace(
            0,
            np.nan,
        )

        data["sender_balance_ratio"] = (
            data["amount"] / sender_balance
        ).fillna(0)

        # --------------------------------------------------
        # Receiver Balance Increase
        # --------------------------------------------------

        data["receiver_balance_change"] = (
            data["newbalanceDest"]
            - data["oldbalanceDest"]
        )

        # --------------------------------------------------
        # Sender Balance Decrease
        # --------------------------------------------------

        data["sender_balance_change"] = (
            data["oldbalanceOrg"]
            - data["newbalanceOrig"]
        )

        # --------------------------------------------------
        # Balance Error
        # (difference between expected and actual)
        # --------------------------------------------------

        expected_sender = (
            data["oldbalanceOrg"]
            - data["amount"]
        )

        data["sender_balance_error"] = (
            expected_sender
            - data["newbalanceOrig"]
        )

        expected_receiver = (
            data["oldbalanceDest"]
            + data["amount"]
        )

        data["receiver_balance_error"] = (
            expected_receiver
            - data["newbalanceDest"]
        )

        # --------------------------------------------------
        # Impossible Balance Flag
        # --------------------------------------------------

        data["balance_anomaly"] = (
            (
                data["sender_balance_error"].abs() > 1
            )
            |
            (
                data["receiver_balance_error"].abs() > 1
            )
        ).astype("int8")

        return data
    