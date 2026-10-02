"""
Athena v1.3
--------------------
Fusion Ensemble

Model-output fusion over already-trained binary classifiers.

Reuses VotingEnsemble's validation and single-pass inference; the only
thing it adds is the fusion rule, so there is no duplicated logic.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from app.ml.ensemble.voting import VotingEnsemble

__all__ = ["FusionEnsemble"]

_METHODS = ("weighted_average", "max")


class FusionEnsemble(VotingEnsemble):
    """
    Probability fusion with a selectable rule.

    Parameters
    ----------
    models, weights, threshold : see ``VotingEnsemble``.
    method : fusion rule.
        ``"weighted_average"`` - weighted mean of fraud probabilities
            (identical to VotingEnsemble; the default).
        ``"max"`` - highest fraud probability among models with non-zero
            weight. Recall-oriented: one confident model is enough to
            raise the score. Expect more false positives, so raise the
            threshold or send VERIFY-band cases to review.

    Notes
    -----
    No meta-model is trained, so no hold-out data is needed (unlike
    ``StackingEnsemble``).
    """

    def __init__(
        self,
        models: Mapping[str, Any],
        weights: Mapping[str, float] | None = None,
        threshold: float = 0.5,
        method: str = "weighted_average",
    ) -> None:
        if method not in _METHODS:
            raise ValueError(f"method must be one of {list(_METHODS)}, got {method!r}.")
        super().__init__(models, weights, threshold)
        self.method = method
        self._active = self._weights > 0  # models excluded from "max" when weight is 0

    def _combine(self, matrix: np.ndarray) -> np.ndarray:
        if self.method == "max":
            return matrix[self._active].max(axis=0)
        return self._weights @ matrix

    def predict_with_details(self, X) -> dict[str, Any]:
        details = super().predict_with_details(X)
        details["fusion_method"] = self.method
        return details