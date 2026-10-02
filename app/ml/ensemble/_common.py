"""
Athena v1.0
--------------------
Shared ensemble helpers (used by voting.py and stacking.py).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

__all__ = ["POSITIVE_CLASS", "validate_models", "validate_threshold", "fraud_probability"]

POSITIVE_CLASS = 1


def validate_models(models: Mapping[str, Any]) -> dict[str, Any]:
    """Require a non-empty mapping of models exposing predict_proba(); return a shallow copy."""
    if not models:
        raise ValueError("At least one model is required.")
    for name, model in models.items():
        if not callable(getattr(model, "predict_proba", None)):
            raise TypeError(f"Model '{name}' must implement predict_proba().")
    return dict(models)


def validate_threshold(threshold: float) -> float:
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("threshold must be a number between 0 and 1.")
    return float(threshold)


def fraud_probability(name: str, model: Any, X) -> np.ndarray:
    """
    Class-1 probabilities from one model, validated.

    The fraud column is located via ``classes_`` instead of assuming
    column 1; estimators without ``classes_`` fall back to column 1.
    """
    proba = np.asarray(model.predict_proba(X))
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError(f"Model '{name}': predict_proba() must return shape (n, >=2).")

    column = POSITIVE_CLASS
    classes = getattr(model, "classes_", None)
    if classes is not None:
        hits = np.flatnonzero(np.asarray(classes) == POSITIVE_CLASS)
        if hits.size != 1:
            raise ValueError(
                f"Model '{name}' was not trained with positive class "
                f"{POSITIVE_CLASS} (classes_={list(classes)})."
            )
        column = int(hits[0])

    fraud = proba[:, column].astype(float, copy=False)
    if not np.isfinite(fraud).all():
        raise ValueError(f"Model '{name}' returned NaN/inf probabilities.")
    return fraud
