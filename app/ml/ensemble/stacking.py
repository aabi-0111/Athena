"""
Athena v1.2
--------------------
Stacking Ensemble

Probability-based stacking over already-trained binary classifiers.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

__all__ = ["StackingEnsemble"]

_POSITIVE_CLASS = 1


class StackingEnsemble:
    """
    Probability-stacking ensemble.

    Base models must already be fitted and implement ``predict_proba()``.
    Their fraud probabilities are the features of a meta-model.

    LEAKAGE WARNING
    ---------------
    ``fit(X, y)`` must receive HOLD-OUT data: rows that none of the base
    models were trained on, and that are not the final test set. Base
    models score their own training rows near-perfectly (Random Forest
    especially), so a meta-model fitted on those rows learns to over-trust
    them and fails on unseen transactions. Carve a validation split out of
    the training data *before* training the base models, fit the base
    models on the remainder, fit this ensemble on the validation split, and
    evaluate on the untouched test set.

    Parameters
    ----------
    models : mapping of name -> fitted model implementing predict_proba().
    threshold : probability cutoff for the binary prediction, in [0, 1].
    meta_model : sklearn-style classifier (default LogisticRegression).
        ``fit()`` trains a clone, so the object you pass is never mutated.
        A pre-fitted meta-model is usable for prediction without calling
        ``fit()``.
    """

    def __init__(
        self,
        models: Mapping[str, Any],
        threshold: float = 0.5,
        meta_model: Any | None = None,
    ) -> None:
        if not models:
            raise ValueError("At least one base model is required.")

        for name, model in models.items():
            if not callable(getattr(model, "predict_proba", None)):
                raise TypeError(f"Model '{name}' must implement predict_proba().")

        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError("threshold must be a number between 0 and 1.")

        self.models = dict(models)
        self.threshold = float(threshold)
        self._names = tuple(self.models)

        self.meta_model = (
            meta_model if meta_model is not None else LogisticRegression(max_iter=1000)
        )
        for method in ("fit", "predict_proba"):
            if not callable(getattr(self.meta_model, method, None)):
                raise TypeError(f"meta_model must implement {method}().")

        # Fitted meta-model (None until fit(), unless one was supplied pre-fitted).
        self.meta_model_ = None
        try:
            check_is_fitted(self.meta_model)
            self.meta_model_ = self.meta_model
        except NotFittedError:
            pass

    # ------------------------------------------------------------------

    @staticmethod
    def _fraud_probability(name: str, model: Any, X) -> np.ndarray:
        """Class-1 probabilities from one model, validated."""
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError(
                f"Model '{name}': predict_proba() must return shape (n, >=2)."
            )

        column = _POSITIVE_CLASS
        classes = getattr(model, "classes_", None)
        if classes is not None:
            hits = np.flatnonzero(np.asarray(classes) == _POSITIVE_CLASS)
            if hits.size != 1:
                raise ValueError(
                    f"Model '{name}' was not trained with positive class "
                    f"{_POSITIVE_CLASS} (classes_={list(classes)})."
                )
            column = int(hits[0])

        fraud = proba[:, column].astype(float, copy=False)
        if not np.isfinite(fraud).all():
            raise ValueError(f"Model '{name}' returned NaN/inf probabilities.")
        return fraud

    def _base_matrix(self, X) -> np.ndarray:
        """Run each base model once -> array of shape (n_samples, n_models)."""
        columns = [self._fraud_probability(n, self.models[n], X) for n in self._names]
        if len({c.shape[0] for c in columns}) != 1:
            raise ValueError("Base models returned different numbers of predictions.")
        return np.column_stack(columns)

    def _meta_proba(self, base_matrix: np.ndarray) -> np.ndarray:
        if self.meta_model_ is None:
            raise RuntimeError("StackingEnsemble must be fitted before prediction.")
        return self._fraud_probability("meta_model", self.meta_model_, base_matrix)

    # ------------------------------------------------------------------

    def fit(self, X, y) -> "StackingEnsemble":
        """
        Fit the meta-model on base-model probabilities.

        ``X``/``y`` must be hold-out data (see the class leakage warning).
        """
        base_matrix = self._base_matrix(X)
        y = np.asarray(y)

        if y.ndim != 1 or y.shape[0] != base_matrix.shape[0]:
            raise ValueError("y must be 1-D with one label per row of X.")
        if base_matrix.shape[0] == 0:
            raise ValueError("Cannot fit stacking ensemble on empty data.")
        if not np.isin(y, (0, 1)).all() or np.unique(y).size < 2:
            raise ValueError("y must be binary (0/1) and contain both classes.")

        self.meta_model_ = clone(self.meta_model).fit(base_matrix, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Meta-model fraud probability per transaction."""
        return self._meta_proba(self._base_matrix(X))

    def predict(self, X) -> np.ndarray:
        """Binary prediction: 0 = legitimate, 1 = fraud."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def predict_with_details(self, X) -> dict[str, Any]:
        """Base-model and stacked outputs from a single inference pass."""
        base_matrix = self._base_matrix(X)
        proba = self._meta_proba(base_matrix)

        return {
            "model_probabilities": dict(zip(self._names, base_matrix.T)),
            "ensemble_probability": proba,
            "ensemble_prediction": (proba >= self.threshold).astype(int),
            "models_used": list(self._names),
            "threshold": self.threshold,
            "meta_model": type(self.meta_model).__name__,
        }