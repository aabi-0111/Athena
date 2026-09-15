"""
Athena v1.0
--------------------
LightGBM Model

Responsibilities
----------------
1. Build and configure a LightGBM classifier.
2. Validate hyperparameters before model creation.
3. Return an untrained LightGBM estimator.
"""

from __future__ import annotations

from typing import Any, Final

from lightgbm import LGBMClassifier

try:
    from app.core.exceptions import ModelFactoryError
except ImportError:
    from core.exceptions import ModelFactoryError

__all__ = ["build_lightgbm_model"]

_VALID_BOOSTING_TYPES: Final[frozenset[str]] = frozenset({"gbdt", "dart", "goss", "rf"})
_VALID_OBJECTIVES: Final[frozenset[str]] = frozenset({"binary", "cross_entropy"})


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


def _validate_non_negative_int(name: str, value: int | None) -> None:
    """
    Validate that an integer hyperparameter is >= 0.

    Used for max_depth, where -1 has the special LightGBM meaning of
    "no depth limit" and must not be rejected.
    """
    if value is not None:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, not bool.")
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value < -1:
            raise ValueError(f"{name} must be >= -1 (-1 means unlimited).")


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

    Used for reg_alpha/reg_lambda/min_split_gain, where 0 is the valid,
    meaningful "no regularization" default.
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
    Validate a probability-like parameter in (0.0, 1.0].
    """
    if value is not None:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be numeric, not bool.")
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")
        if not 0.0 < float(value) <= 1.0:
            raise ValueError(f"{name} must be in (0.0, 1.0].")


def build_lightgbm_model(
    *,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = -1,
    num_leaves: int = 31,
    min_child_samples: int = 20,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    min_split_gain: float = 0.0,
    boosting_type: str = "gbdt",
    objective: str = "binary",
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
    verbosity: int = -1,
    **kwargs: Any,
) -> LGBMClassifier:
    """
    Build a configured LightGBM classifier.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds. Must be > 0.
    learning_rate : float
        Learning rate. Must be > 0.
    max_depth : int
        Maximum tree depth. Must be >= -1 (-1 means no limit, LightGBM's
        default and the convention this builder follows).
    num_leaves : int
        Maximum number of leaves per tree. Must be > 0. LightGBM grows
        leaf-wise rather than depth-wise, so this — not max_depth — is
        the primary complexity control; kept independently validated
        rather than derived from max_depth.
    min_child_samples : int
        Minimum number of data points required in a leaf. Must be > 0.
    subsample : float
        Row sampling ratio (bagging_fraction), in (0.0, 1.0].
    colsample_bytree : float
        Feature sampling ratio (feature_fraction), in (0.0, 1.0].
    reg_alpha : float
        L1 regularization. Must be >= 0.
    reg_lambda : float
        L2 regularization. Must be >= 0.
    min_split_gain : float
        Minimum loss reduction required for a split. Must be >= 0.
    boosting_type : str
        One of {'gbdt', 'dart', 'goss', 'rf'}.
    objective : str
        One of {'binary', 'cross_entropy'} (not validated against the
        full LightGBM objective list; these two are the meaningful
        options for this binary fraud-detection use case).
    class_weight : str | dict | None
        Class weighting strategy. 'balanced' by default, matching the
        random_forest builder's handling of PaySim's severe imbalance.
    random_state : int
        Random seed for reproducibility.
    n_jobs : int
        Number of CPU cores.
    verbosity : int
        LightGBM log verbosity. Defaults to -1 (silent) since per-tree
        LightGBM logging is disproportionately noisy on multi-million-row
        PaySim training runs.

    Returns
    -------
    LGBMClassifier
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
        _validate_positive_int("num_leaves", num_leaves)
        _validate_positive_int("min_child_samples", min_child_samples)

        _validate_positive_float("learning_rate", learning_rate)
        _validate_non_negative_float("reg_alpha", reg_alpha)
        _validate_non_negative_float("reg_lambda", reg_lambda)
        _validate_non_negative_float("min_split_gain", min_split_gain)

        _validate_probability("subsample", subsample)
        _validate_probability("colsample_bytree", colsample_bytree)

        if boosting_type not in _VALID_BOOSTING_TYPES:
            raise ValueError(
                f"boosting_type must be one of {sorted(_VALID_BOOSTING_TYPES)}, "
                f"got {boosting_type!r}."
            )
        if objective not in _VALID_OBJECTIVES:
            raise ValueError(
                f"objective must be one of {sorted(_VALID_OBJECTIVES)}, "
                f"got {objective!r}."
            )
        if boosting_type == "rf" and (subsample >= 1.0 or colsample_bytree >= 1.0):
            # LightGBM requires bagging_fraction/feature_fraction < 1.0 in
            # 'rf' boosting mode; surface this here with a clear message
            # instead of the underlying LightGBM C++ error.
            raise ValueError(
                "boosting_type='rf' requires subsample < 1.0 and "
                "colsample_bytree < 1.0."
            )
    except (TypeError, ValueError) as exc:
        raise ModelFactoryError(f"Invalid lightgbm hyperparameters: {exc}") from exc

    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_split_gain=min_split_gain,
        boosting_type=boosting_type,
        objective=objective,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=verbosity,
        **kwargs,
    )
