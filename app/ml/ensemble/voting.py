"""
Athena v1.1
--------------------
Voting Ensemble

Weighted soft-voting over already-trained binary classifiers.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

__all__ = ["VotingEnsemble"]

_POSITIVE_CLASS = 1


class VotingEnsemble:
    """
    Weighted probability-averaging ensemble.

    Parameters
    ----------
    models : mapping of name -> fitted model implementing predict_proba().
    weights : optional mapping of name -> non-negative weight, covering
        exactly the same names as ``models``. Normalized internally, so
        only the ratios matter. Defaults to equal weights.
    threshold : probability cutoff for the binary prediction, in [0, 1].
    """

    def __init__(
        self,
        models: Mapping[str, Any],
        weights: Mapping[str, float] | None = None,
        threshold: float = 0.5,
    ) -> None:
        if not models:
            raise ValueError("At least one model is required.")

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

        if weights is None:
            raw = np.ones(len(self._names))
        else:
            if set(weights) != set(self.models):
                raise ValueError(
                    "Weights must be provided for exactly the models in the ensemble."
                )
            raw = np.array([weights[n] for n in self._names], dtype=float)
            if not np.isfinite(raw).all() or (raw < 0).any():
                raise ValueError("Model weights must be finite and non-negative.")
            if raw.sum() == 0:
                raise ValueError("At least one model weight must be greater than zero.")

        self._weights = raw / raw.sum()

    @property
    def weights(self) -> dict[str, float]:
        """Normalized weights (sum to 1)."""
        return dict(zip(self._names, self._weights.tolist()))

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

    def _model_matrix(self, X) -> np.ndarray:
        """Run each model once -> array of shape (n_models, n_samples)."""
        rows = [self._fraud_probability(n, self.models[n], X) for n in self._names]
        if len({r.shape[0] for r in rows}) != 1:
            raise ValueError("Models returned different numbers of predictions.")
        return np.vstack(rows)

    def _combine(self, matrix: np.ndarray) -> np.ndarray:
        """Collapse (n_models, n_samples) into (n_samples,). Override to change the rule."""
        return self._weights @ matrix

    def predict_proba(self, X) -> np.ndarray:
        """Weighted ensemble fraud probability per transaction."""
        return self._combine(self._model_matrix(X))

    def predict(self, X) -> np.ndarray:
        """Binary prediction: 0 = legitimate, 1 = fraud."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def predict_with_details(self, X) -> dict[str, Any]:
        """Ensemble output plus per-model probabilities (one inference pass)."""
        matrix = self._model_matrix(X)
        ensemble = self._combine(matrix)

        return {
            "model_probabilities": dict(zip(self._names, matrix)),
            "ensemble_probability": ensemble,
            "ensemble_prediction": (ensemble >= self.threshold).astype(int),
            "models_used": list(self._names),
            "weights": self.weights,
            "threshold": self.threshold,
        }