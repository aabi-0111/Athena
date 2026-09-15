"""
Athena v1.0
--------------------
Random Forest Model

Responsibilities
----------------
1. Build and configure a Random Forest classifier.
2. Validate hyperparameters before model creation.
3. Return an untrained sklearn estimator.
"""

from __future__ import annotations

from typing import Any, Final

from sklearn.ensemble import RandomForestClassifier

from app.core.exceptions import ModelFactoryError

__all__ = ["build_random_forest"]

_VALID_CRITERIA: Final[frozenset[str]] = frozenset({"gini", "entropy", "log_loss"})


def _validate_int_or_unit_float(name: str, value: int | float | None, *, int_minimum: int = 1) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int or float, not bool.")
    if isinstance(value, int):
        if value < int_minimum:
            raise ValueError(f"{name} must be an integer >= {int_minimum}.")
        return
    if isinstance(value, float):
        if not (0.0 < value <= 1.0):
            raise ValueError(f"{name} must be a float in (0.0, 1.0].")
        return
    raise TypeError(f"{name} must be an int or float, got {type(value).__name__}.")


def _validate_positive_int(name: str, value: int | None) -> None:
    if value is not None:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, not bool.")
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0.")


def build_random_forest(
    *,
    n_estimators: int = 300,
    criterion: str = "gini",
    max_depth: int | None = None,
    min_samples_split: int | float = 2,
    min_samples_leaf: int | float = 1,
    max_features: str | int | float | None = "sqrt",
    bootstrap: bool = True,
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs: Any,
) -> RandomForestClassifier:
    try:
        _validate_positive_int("n_estimators", n_estimators)
        if max_depth is not None:
            _validate_positive_int("max_depth", max_depth)
        _validate_int_or_unit_float("min_samples_split", min_samples_split, int_minimum=2)
        _validate_int_or_unit_float("min_samples_leaf", min_samples_leaf, int_minimum=1)

        if criterion not in _VALID_CRITERIA:
            raise ValueError(f"criterion must be one of {sorted(_VALID_CRITERIA)}, got {criterion!r}.")

        if isinstance(max_features, bool):
            raise TypeError("max_features must be a str, int, float, or None, not bool.")
        if max_features is not None and not isinstance(max_features, str):
            _validate_int_or_unit_float("max_features", max_features, int_minimum=1)

        if not isinstance(bootstrap, bool):
            raise TypeError("bootstrap must be a bool.")
        if kwargs.get("oob_score") and not bootstrap:
            raise ValueError("oob_score=True requires bootstrap=True.")
    except (TypeError, ValueError) as exc:
        raise ModelFactoryError(f"Invalid random_forest hyperparameters: {exc}") from exc

    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )
