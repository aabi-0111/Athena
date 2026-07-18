"""
Athena v1.2
--------------------
Time Feature Engineering
Responsibilities
----------------
1. Hour extraction
2. Day extraction
3. Week extraction
4. Time-of-day encoding
5. Weekend flag
6. Cyclic time encoding
Author: Athena
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("step",)

_FLAG_DTYPE = "int8"
_FLOAT_DTYPE = "float32"

# Time-of-day bucket labels, indexed by (hour // 6): 0=night, 1=morning, 2=afternoon, 3=evening
_TIME_OF_DAY_LABELS = np.array(["night", "morning", "afternoon", "evening"], dtype=object)


class TimeFeatureError(ValueError):
    """Raised when required columns are missing for time feature engineering."""


class TimeFeatures:
    """
    Generates time-based features from the PaySim simulation step column.

    PaySim:
        1 step = 1 hour
    """

    HOURS_PER_DAY = 24
    DAYS_PER_WEEK = 7
    HOURS_PER_BUCKET = 6  # night/morning/afternoon/evening are fixed 6h blocks

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate(df)
        data = df.copy()

        step = data["step"].to_numpy(dtype="int32", copy=False)

        # ==================================================
        # Hour / day / day-of-week / week (single integer-arithmetic pass)
        # ==================================================
        hour = step % self.HOURS_PER_DAY
        day = step // self.HOURS_PER_DAY
        day_of_week = day % self.DAYS_PER_WEEK
        week = day // self.DAYS_PER_WEEK

        data["hour"] = hour.astype(_FLAG_DTYPE)
        data["day"] = day.astype("int16")
        data["day_of_week"] = day_of_week.astype(_FLAG_DTYPE)
        data["week"] = week.astype("int16")

        # ==================================================
        # Weekend flag
        # ==================================================
        data["is_weekend"] = (day_of_week >= 5).astype(_FLAG_DTYPE)

        # ==================================================
        # Time-of-day bucket (single integer division, no comparisons needed
        # since night/morning/afternoon/evening are fixed 6-hour blocks)
        # ==================================================
        bucket = hour // self.HOURS_PER_BUCKET  # 0..3

        data["is_night"] = (bucket == 0).astype(_FLAG_DTYPE)
        data["is_morning"] = (bucket == 1).astype(_FLAG_DTYPE)
        data["is_afternoon"] = (bucket == 2).astype(_FLAG_DTYPE)
        data["is_evening"] = (bucket == 3).astype(_FLAG_DTYPE)

        # ==================================================
        # Cyclic hour encoding
        # ==================================================
        hour_angle = (2 * np.pi * hour / self.HOURS_PER_DAY).astype(_FLOAT_DTYPE)
        data["hour_sin"] = np.sin(hour_angle).astype(_FLOAT_DTYPE)
        data["hour_cos"] = np.cos(hour_angle).astype(_FLOAT_DTYPE)

        # ==================================================
        # Cyclic day encoding
        # ==================================================
        day_angle = (2 * np.pi * day_of_week / self.DAYS_PER_WEEK).astype(_FLOAT_DTYPE)
        data["day_sin"] = np.sin(day_angle).astype(_FLOAT_DTYPE)
        data["day_cos"] = np.cos(day_angle).astype(_FLOAT_DTYPE)

        return data

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise TimeFeatureError(
                f"Missing required column(s) for time feature engineering: {missing}"
            )
        if (df["step"] < 0).any():
            raise TimeFeatureError("'step' contains negative values; expected non-negative simulation steps.")
        