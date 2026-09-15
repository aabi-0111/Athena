"""
Athena v1.0
--------------------
Feature Importance

Responsibilities
----------------
1. Extract feature importance from supported models.
2. Return feature rankings as a DataFrame.
3. Optionally save feature importance to CSV.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

__all__ = ["FeatureImportanceExtractor", "FeatureImportanceError"]

_FEATURE_COL: Final[str] = "feature"
_IMPORTANCE_COL: Final[str] = "importance"
_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset({_FEATURE_COL, _IMPORTANCE_COL})


class FeatureImportanceError(RuntimeError):
    """Raised when feature importance cannot be extracted or saved."""


class FeatureImportanceExtractor:
    """
    Extract feature importance from trained tree-based models.
    """

    @staticmethod
    def extract(
        model: BaseEstimator,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """
        Extract feature importance.

        Parameters
        ----------
        model : BaseEstimator
            Trained model supporting feature_importances_.
        feature_names : list[str]
            Names of input features, in the same order the model was
            trained on. Must be unique — duplicate names make it
            impossible to tell which physical feature a given
            importance row refers to, which is unacceptable for an
            artifact that may feed SHAP/regulatory review.

        Returns
        -------
        pd.DataFrame
            Two columns, 'feature' and 'importance', sorted descending
            by importance.

        Raises
        ------
        FeatureImportanceError
            If the model doesn't expose feature_importances_, if
            feature_names is empty, contains duplicates, or doesn't
            match the importance array in length, or if the extracted
            importances contain NaN/Inf values.
        """
        if not hasattr(model, "feature_importances_"):
            raise FeatureImportanceError(
                f"{type(model).__name__} does not expose "
                "'feature_importances_'. Ensure the model is fitted "
                "and is a tree-based estimator."
            )
        if not feature_names:
            raise FeatureImportanceError("feature_names must not be empty.")
        if len(feature_names) != len(set(feature_names)):
            seen: set[str] = set()
            duplicates = sorted(
                {name for name in feature_names if name in seen or seen.add(name)}
            )
            raise FeatureImportanceError(
                f"feature_names contains duplicates: {duplicates}. Duplicate "
                "names make importance rows ambiguous."
            )

        importance = np.asarray(model.feature_importances_)
        if importance.ndim != 1:
            raise FeatureImportanceError(
                f"feature_importances_ must be 1-dimensional, got shape "
                f"{importance.shape}."
            )
        if len(feature_names) != len(importance):
            raise FeatureImportanceError(
                f"feature_names length ({len(feature_names)}) does not match "
                f"number of feature importances ({len(importance)})."
            )
        if not np.all(np.isfinite(importance)):
            raise FeatureImportanceError(
                f"{type(model).__name__} produced NaN or infinite feature "
                "importances. This can happen with some boosting "
                "configurations when a feature is never used in any split; "
                "check the model's training data and hyperparameters."
            )

        df = pd.DataFrame(
            {
                _FEATURE_COL: feature_names,
                _IMPORTANCE_COL: importance,
            }
        ).sort_values(
            by=_IMPORTANCE_COL,
            ascending=False,
            ignore_index=True,
        )
        return df

    @staticmethod
    def top_features(
        importance_df: pd.DataFrame,
        n: int = 20,
    ) -> pd.DataFrame:
        """
        Return the top N most important features.

        Parameters
        ----------
        importance_df : pd.DataFrame
            Output of extract(), or any DataFrame with 'feature' and
            'importance' columns.
        n : int
            Number of features to return.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        ValueError
            If n <= 0.
        FeatureImportanceError
            If importance_df is missing the expected columns.
        """
        if n <= 0:
            raise ValueError("n must be greater than 0.")
        missing = _REQUIRED_COLUMNS - set(importance_df.columns)
        if missing:
            raise FeatureImportanceError(
                f"importance_df is missing required column(s): {sorted(missing)}. "
                "Expected output from FeatureImportanceExtractor.extract()."
            )
        return importance_df.head(n).copy()

    @staticmethod
    def save_csv(
        importance_df: pd.DataFrame,
        output_path: str | Path,
    ) -> Path:
        """
        Save feature importance as CSV.

        Writes atomically: content is written to a temporary file in
        the destination directory, then moved into place with
        os.replace, so a failure or interruption mid-write never
        leaves a corrupt or partial CSV at `output_path`.

        Parameters
        ----------
        importance_df : pd.DataFrame
        output_path : str | Path

        Returns
        -------
        Path
            Saved CSV path.

        Raises
        ------
        FeatureImportanceError
            If importance_df is missing the expected columns.
        """
        missing = _REQUIRED_COLUMNS - set(importance_df.columns)
        if missing:
            raise FeatureImportanceError(
                f"importance_df is missing required column(s): {sorted(missing)}. "
                "Expected output from FeatureImportanceExtractor.extract()."
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path_str = tempfile.mkstemp(
            dir=output_path.parent, prefix=".tmp-", suffix=".csv"
        )
        os.close(fd)
        tmp_path = Path(tmp_path_str)
        try:
            importance_df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, output_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        return output_path
    