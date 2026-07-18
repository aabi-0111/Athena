"""
Athena v1.0
--------------------
Evaluation Metrics

Responsibilities
----------------
1. Compute classification metrics.
2. Validate prediction inputs.
3. Return all metrics in a structured dictionary.
"""

from __future__ import annotations

import warnings
from typing import Any, Final

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = ["MetricsCalculator", "MetricsError"]

# Metrics keys that are None-able because they require y_prob and/or a
# second class to be present in y_true.
_PROB_DEPENDENT_KEYS: Final[frozenset[str]] = frozenset({"roc_auc", "pr_auc"})


class MetricsError(ValueError):
    """Raised when evaluation metrics cannot be computed."""


class MetricsCalculator:
    """
    Computes evaluation metrics for binary classification models.

    Notes
    -----
    `precision`, `recall`, `f1_score`, and the classification report are
    computed from `y_pred` — i.e. already-thresholded predicted labels,
    not `y_prob`. To evaluate at a different decision threshold, derive
    a new `y_pred` at that threshold before calling `compute()`.

    `roc_auc` and `pr_auc` are set to `None` (rather than `nan` or a
    misleading default) when they can't be meaningfully computed — most
    commonly when `y_true` contains only one class, which is a real
    possibility on small or time-windowed evaluation slices given how
    rare the positive (fraud) class is.
    """

    @staticmethod
    def _validate_inputs(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Validate and normalize prediction arrays.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (y_true, y_pred) coerced to 1-D numpy arrays.

        Raises
        ------
        MetricsError
            If arrays are empty, mismatched in length, not 1-D, contain
            NaN/Inf, or contain more than two distinct classes.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.ndim != 1:
            raise MetricsError(
                f"y_true must be 1-dimensional, got shape {y_true.shape}."
            )
        if y_pred.ndim != 1:
            raise MetricsError(
                f"y_pred must be 1-dimensional, got shape {y_pred.shape}."
            )
        if y_true.size == 0:
            raise MetricsError("y_true is empty.")
        if y_pred.size == 0:
            raise MetricsError("y_pred is empty.")
        if y_true.size != y_pred.size:
            raise MetricsError(
                f"y_true and y_pred must have the same length "
                f"(got {y_true.size} and {y_pred.size})."
            )
        if np.issubdtype(y_true.dtype, np.floating) and not np.all(np.isfinite(y_true)):
            raise MetricsError("y_true contains NaN or infinite values.")
        if np.issubdtype(y_pred.dtype, np.floating) and not np.all(np.isfinite(y_pred)):
            raise MetricsError("y_pred contains NaN or infinite values.")

        n_true_classes = len(np.unique(y_true))
        n_pred_classes = len(np.unique(y_pred))
        if n_true_classes > 2:
            raise MetricsError(
                f"y_true must be binary; found {n_true_classes} distinct "
                f"classes: {sorted(np.unique(y_true).tolist())}."
            )
        if n_pred_classes > 2:
            raise MetricsError(
                f"y_pred must be binary; found {n_pred_classes} distinct "
                f"classes: {sorted(np.unique(y_pred).tolist())}."
            )

        return y_true, y_pred

    @staticmethod
    def _validate_prob(y_prob: np.ndarray, expected_len: int) -> np.ndarray:
        """
        Validate and normalize the predicted-probability array.
        """
        y_prob = np.asarray(y_prob)
        if y_prob.ndim != 1:
            raise MetricsError(
                f"y_prob must be 1-dimensional, got shape {y_prob.shape}."
            )
        if y_prob.size != expected_len:
            raise MetricsError(
                f"y_prob must have the same length as y_true "
                f"(got {y_prob.size}, expected {expected_len})."
            )
        if not np.all(np.isfinite(y_prob)):
            raise MetricsError("y_prob contains NaN or infinite values.")
        return y_prob

    @classmethod
    def compute(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Compute evaluation metrics.

        Parameters
        ----------
        y_true : ndarray
            Ground-truth binary labels.
        y_pred : ndarray
            Predicted binary labels.
        y_prob : ndarray | None
            Predicted probabilities for the positive class.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics. `roc_auc` and
            `pr_auc` are `None` if `y_prob` was not provided, or if
            `y_true` contains only one class (these metrics are
            undefined in that case rather than a misleading number).

        Raises
        ------
        MetricsError
            If inputs are empty, mismatched in length, non-binary,
            non-1D, or contain NaN/Inf values.
        """
        y_true, y_pred = cls._validate_inputs(y_true, y_pred)
        single_class = len(np.unique(y_true)) < 2

        results: dict[str, Any] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            ),
        }

        if y_prob is None:
            results["roc_auc"] = None
            results["pr_auc"] = None
            return results

        y_prob = cls._validate_prob(y_prob, expected_len=y_true.size)

        if single_class:
            # roc_auc_score and precision_recall_curve are undefined with
            # only one class present in y_true; some sklearn versions
            # raise, others return nan with a warning. Either way the
            # value isn't meaningful, so report it explicitly as None
            # instead of propagating nan or a spurious warning.
            results["roc_auc"] = None
            results["pr_auc"] = None
            return results

        with warnings.catch_warnings():
            warnings.simplefilter("error", category=Warning)
            try:
                results["roc_auc"] = roc_auc_score(y_true, y_prob)
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
                results["pr_auc"] = auc(recall_curve, precision_curve)
            except Exception as exc:
                raise MetricsError(
                    f"Failed to compute probability-based metrics: {exc}"
                ) from exc

        return results
    