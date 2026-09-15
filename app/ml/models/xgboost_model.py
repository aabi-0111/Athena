"""
Athena v1.0
--------------------
XGBoost Model

Responsibilities
----------------
1. Build and configure an XGBoost classifier.
2. Validate hyperparameters before model creation.
3. Return an untrained XGBoost estimator.
"""

from __future__ import annotations

from typing import Any, Final

from xgboost import XGBClassifier

from app.core.exceptions import ModelFactoryError

__all__ = ["build_xgboost_model"]

_VALID_TREE_METHODS: Final[frozenset[str]] = frozenset({"auto", "exact", "approx", "hist", "gpu_hist"})


def _validate_non_negative_int(name: str, value: int | None) -> None:
    if value is not None:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, not bool.")
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value < 0:
            raise ValueError(f"{name} must be >= 0.")


def _validate_positive_int(name: str, value: int | None) -> None:
    if value is not None:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, not bool.")
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0.")


def _validate_positive_float(name: str, value: float | None) -> None:
    if value is not None:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be numeric, not bool.")
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0.")


def _validate_non_negative_float(name: str, value: float | None) -> None:
    if value is not None:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be numeric, not bool.")
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")
        if value < 0:
            raise ValueError(f"{name} must be >= 0.")


def _validate_probability(name: str, value: float | None) -> None:
    if value is not None:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be numeric, not bool.")
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")


def build_xgboost_model(
    *,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    min_child_weight: int = 1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    gamma: float = 0.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    scale_pos_weight: float = 1.0,
    objective: str = "binary:logistic",
    eval_metric: str = "logloss",
    random_state: int = 42,
    n_jobs: int = -1,
    tree_method: str = "hist",
    verbosity: int = 0,
    **kwargs: Any,
) -> XGBClassifier:
    try:
        _validate_positive_int("n_estimators", n_estimators)
        _validate_non_negative_int("max_depth", max_depth)
        _validate_positive_int("min_child_weight", min_child_weight)
        _validate_positive_float("learning_rate", learning_rate)
        _validate_non_negative_float("gamma", gamma)
        _validate_non_negative_float("reg_alpha", reg_alpha)
        _validate_positive_float("reg_lambda", reg_lambda)
        _validate_positive_float("scale_pos_weight", scale_pos_weight)
        _validate_probability("subsample", subsample)
        _validate_probability("colsample_bytree", colsample_bytree)

        if tree_method not in _VALID_TREE_METHODS:
            raise ValueError(
                f"tree_method must be one of {sorted(_VALID_TREE_METHODS)}, got {tree_method!r}."
            )
    except (TypeError, ValueError) as exc:
        raise ModelFactoryError(f"Invalid xgboost hyperparameters: {exc}") from exc

    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        objective=objective,
        eval_metric=eval_metric,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        verbosity=verbosity,
        **kwargs,
    )
