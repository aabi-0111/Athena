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

try:
    from core.exceptions import ModelFactoryError
except ImportError:
    try:
        from app.core.exceptions import ModelFactoryError
    except ImportError:
        class ModelFactoryError(ValueError):
            """Fallback if core.exceptions is unavailable (standalone use)."""

__all__ = ["build_xgboost_model"]

# XGBoost's supported tree construction algorithms as of 2.x/3.x.
# "auto"/"exact"/"approx" retained for CPU paths; "gpu_hist" removed in
# XGBoost 2.0+ in favor of device="cuda" + tree_method="hist", but is
# still accepted here defensively since some deployments may pin older
# XGBoost versions.
_VALID_TREE_METHODS: Final[frozenset[str]] = frozenset(
    {"auto", "exact", "approx", "hist", "gpu_hist"}
)


def _validate_non_negative_int(name: str, value: int | None) -> None:
    """
    Validate that an integer hyperparameter is >= 0 (not just > 0).

    Used for max_depth, where 0 has the special XGBoost meaning of
    "no depth limit" and must not be rejected.
    """
    if value is not None:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, not bool.")
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value < 0:
            raise ValueError(f"{name} must be >= 0.")


def _validate_positive_int(name: str, value: int | None) -> None:
    """
    Validate that an integer hyperparameter is a positive, non-bool int.
    """
    if value is not None:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, not bool.")
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0.")


def _validate_positive_float(name: str, value: float | None) -> None:
    """
    Validate that a numeric hyperparameter is strictly positive.
    """
    if value is not None:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be numeric, not bool.")
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0.")


def _validate_non_negative_float(name: str, value: float | None) -> None:
    """
    Validate that a numeric hyperparameter is >= 0.

    Used for gamma and reg_alpha, where 0 is the valid, meaningful
    "no regularization / no split-loss threshold" default.
    """
    if value is not None:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be numeric, not bool.")
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")
        if value < 0:
            raise ValueError(f"{name} must be >= 0.")


def _validate_probability(name: str, value: float | None) -> None:
    """
    Validate a probability-like parameter in [0.0, 1.0].
    """
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
    """
    Build a configured XGBoost classifier.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds. Must be > 0.
    learning_rate : float
        Learning rate. Must be > 0.
    max_depth : int
        Maximum depth of trees. Must be >= 0 (0 means "no limit").
    min_child_weight : int
        Minimum child weight. Must be > 0.
    subsample : float
        Row sampling ratio, in [0.0, 1.0].
    colsample_bytree : float
        Feature sampling ratio, in [0.0, 1.0].
    gamma : float
        Minimum loss reduction required for split. Must be >= 0.
    reg_alpha : float
        L1 regularization. Must be >= 0.
    reg_lambda : float
        L2 regularization. Must be > 0.
    scale_pos_weight : float
        Positive class weighting. Must be > 0.
    objective : str
        Learning objective (not validated against an allowlist; XGBoost
        supports many valid values and will raise its own clear error
        for unrecognized ones).
    eval_metric : str
        Evaluation metric (same rationale as `objective`).
    random_state : int
        Random seed.
    n_jobs : int
        Number of CPU cores.
    tree_method : str
        Tree construction algorithm. One of {'auto', 'exact', 'approx',
        'hist', 'gpu_hist'}.
    verbosity : int
        XGBoost log verbosity (0-3).

    Returns
    -------
    XGBClassifier
        Untrained classifier.

    Raises
    ------
    ModelFactoryError
        If hyperparameters are invalid or inconsistent (wraps both
        value and type errors so callers, e.g. ModelFactory, can catch
        a single exception type).
    """
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
                f"tree_method must be one of {sorted(_VALID_TREE_METHODS)}, "
                f"got {tree_method!r}."
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
