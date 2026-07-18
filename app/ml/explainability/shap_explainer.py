"""
Athena v1.1
--------------------
SHAP Explainer

Responsibilities
----------------
1. Generate SHAP values.
2. Produce SHAP summary plots.
3. Save SHAP visualizations safely.
4. Support Random Forest, XGBoost and other tree-based models.
5. Validate inputs and provide meaningful exceptions.

Author: Athena
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

__all__ = [
    "SHAPExplainer",
    "SHAPExplainerError",
]


class SHAPExplainerError(RuntimeError):
    """Raised when SHAP explanations cannot be generated."""


class SHAPExplainer:
    """
    SHAP explanation utility for tree-based models.
    """

    def __init__(self, model: BaseEstimator):

        if model is None:
            raise SHAPExplainerError("Model cannot be None.")

        if not isinstance(model, BaseEstimator):
            raise SHAPExplainerError(
                f"Expected sklearn estimator, got {type(model).__name__}."
            )

        try:
            check_is_fitted(model)
        except Exception as exc:
            raise SHAPExplainerError(
                "Model must be fitted before generating SHAP values."
            ) from exc

        self.model = model

        try:
            self.explainer = shap.TreeExplainer(model)
        except Exception as exc:
            raise SHAPExplainerError(
                f"Unable to initialize SHAP TreeExplainer: {exc}"
            ) from exc

    def _validate_dataframe(
        self,
        X: pd.DataFrame,
    ) -> None:

        if not isinstance(X, pd.DataFrame):
            raise SHAPExplainerError(
                f"Expected pandas DataFrame, got {type(X).__name__}."
            )

        if X.empty:
            raise SHAPExplainerError("Input dataframe is empty.")

        if X.columns.duplicated().any():
            duplicates = (
                X.columns[X.columns.duplicated()]
                .unique()
                .tolist()
            )
            raise SHAPExplainerError(
                f"Duplicate feature names detected: {duplicates}"
            )

        if X.isnull().values.any():
            raise SHAPExplainerError(
                "Input dataframe contains NaN values."
            )

    def compute_shap_values(
        self,
        X: pd.DataFrame,
    ):

        self._validate_dataframe(X)

        try:

            values = self.explainer(X)

            if hasattr(values, "values"):
                values = values.values

            return values

        except Exception:

            try:

                values = self.explainer.shap_values(X)

                if isinstance(values, list):
                    values = values[1]

                return values

            except Exception as exc:

                raise SHAPExplainerError(
                    f"Failed to compute SHAP values: {exc}"
                ) from exc

    def summary_plot(
        self,
        X: pd.DataFrame,
        save_path: str | Path,
        *,
        dpi: int = 300,
        plot_size: tuple[int, int] = (10, 8),
    ):

        values = self.compute_shap_values(X)

        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.figure(figsize=plot_size)

        shap.summary_plot(
            values,
            X,
            show=False,
        )

        fd, tmp_name = tempfile.mkstemp(
            dir=save_path.parent,
            prefix=".tmp-",
            suffix=".png",
        )

        os.close(fd)

        tmp_path = Path(tmp_name)

        try:

            plt.savefig(
                tmp_path,
                dpi=dpi,
                bbox_inches="tight",
            )

            os.replace(
                tmp_path,
                save_path,
            )

        except Exception as exc:

            tmp_path.unlink(missing_ok=True)

            raise SHAPExplainerError(
                f"Failed to save SHAP plot: {exc}"
            ) from exc

        finally:

            plt.close()

        logger.info(
            "SHAP summary saved to %s",
            save_path,
        )

        return values
    