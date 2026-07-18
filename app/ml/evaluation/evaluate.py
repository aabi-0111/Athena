"""
Athena v1.0
--------------------
Model Evaluation

Responsibilities
----------------
1. Generate predictions.
2. Compute evaluation metrics.
3. Compute confusion matrix.
4. Extract feature importance.
5. Return a consolidated evaluation report.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator

from app.ml.evaluation.confusion_metrix import (
    ConfusionMatrixError,
    ConfusionMatrixGenerator,
)

from app.ml.evaluation.feature_importance import (
    FeatureImportanceError,
    FeatureImportanceExtractor,
)
from app.ml.evaluation.metrics import MetricsCalculator, MetricsError

__all__ = ["ModelEvaluator", "EvaluationError"]

logger = logging.getLogger(__name__)


class EvaluationError(RuntimeError):
    """Raised when model evaluation fails.

    Wraps every failure that can occur in evaluate() — prediction,
    metrics computation, confusion matrix computation, and feature
    importance extraction — so callers only need to catch this one
    type, with `__cause__` preserving the original exception for
    debugging.
    """


class ModelEvaluator:
    """
    End-to-end evaluation pipeline.
    """

    @staticmethod
    def evaluate(
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, Any]:
        """
        Evaluate a trained model.

        Parameters
        ----------
        model : BaseEstimator
            Trained model.
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            Ground-truth labels.

        Returns
        -------
        dict
            Complete evaluation report with keys: 'metrics',
            'confusion_matrix', 'feature_importance', 'predictions',
            'probabilities'. 'feature_importance' and 'probabilities'
            are None when the model doesn't support them.

        Raises
        ------
        EvaluationError
            If validation, prediction, metrics computation, confusion
            matrix computation, or feature importance extraction
            fails. The original exception is available via
            `__cause__`.
        """
        if not isinstance(X_test, pd.DataFrame):
            raise EvaluationError(
                f"X_test must be a pandas DataFrame, got {type(X_test).__name__}."
            )
        if not isinstance(y_test, pd.Series):
            raise EvaluationError(
                f"y_test must be a pandas Series, got {type(y_test).__name__}."
            )
        if X_test.empty:
            raise EvaluationError("X_test is empty.")
        if y_test.empty:
            raise EvaluationError("y_test is empty.")
        if len(X_test) != len(y_test):
            raise EvaluationError(
                f"X_test and y_test must have the same number of rows "
                f"(got {len(X_test)} and {len(y_test)})."
            )
        if not X_test.index.equals(y_test.index):
            raise EvaluationError(
                "X_test and y_test indices do not match. Predictions are "
                "paired positionally with y_test, so misaligned indices "
                "silently score against the wrong labels. Reset or align "
                "indices before evaluating (e.g. "
                "X_test.reset_index(drop=True), "
                "y_test.reset_index(drop=True))."
            )

        try:
            y_pred = model.predict(X_test)
        except Exception as exc:
            raise EvaluationError(f"Prediction failed: {exc}") from exc

        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)
                if proba.ndim != 2 or proba.shape[1] < 2:
                    raise ValueError(
                        f"predict_proba returned shape {proba.shape}; "
                        "expected (n_samples, >=2) for binary classification."
                    )
                y_prob = proba[:, 1]
            except Exception as exc:
                logger.warning(
                    "predict_proba failed for %s; continuing without "
                    "probability-based metrics (roc_auc, pr_auc). Reason: %s",
                    type(model).__name__,
                    exc,
                )
                y_prob = None

        y_true_arr = y_test.to_numpy()

        try:
            metrics = MetricsCalculator.compute(
                y_true=y_true_arr,
                y_pred=y_pred,
                y_prob=y_prob,
            )
        except MetricsError as exc:
            raise EvaluationError(f"Metrics computation failed: {exc}") from exc

        try:
            confusion = ConfusionMatrixGenerator.compute(
                y_true=y_true_arr,
                y_pred=y_pred,
            )
        except ConfusionMatrixError as exc:
            raise EvaluationError(f"Confusion matrix computation failed: {exc}") from exc

        feature_importance = None
        if hasattr(model, "feature_importances_"):
            try:
                feature_importance = FeatureImportanceExtractor.extract(
                    model=model,
                    feature_names=list(X_test.columns),
                )
            except FeatureImportanceError as exc:
                raise EvaluationError(
                    f"Feature importance extraction failed: {exc}"
                ) from exc

        return {
            "metrics": metrics,
            "confusion_matrix": confusion,
            "feature_importance": feature_importance,
            "predictions": y_pred,
            "probabilities": y_prob,
        }
    