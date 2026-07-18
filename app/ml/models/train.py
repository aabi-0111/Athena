"""
Athena v1.0
--------------------
Model Trainer

Responsibilities
----------------
1. Train a machine-learning model.
2. Validate training data.
3. Measure training time.
4. Return the trained estimator.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from app.ml.models.model_factory import ModelFactory

__all__ = ["ModelTrainer", "ModelTrainingError"]

logger = logging.getLogger(__name__)


class ModelTrainingError(RuntimeError):
    """Raised when model training fails, or when the trainer is given
    an unusable model or malformed training data."""


class ModelTrainer:
    """
    Generic model trainer.

    This class is model-agnostic and can train any estimator
    implementing the standard scikit-learn fit() interface.

    Notes
    -----
    `training_time` and `is_trained` reflect the outcome of the most
    recent `train()` call. If training fails, `training_time` is reset
    to `None` rather than left holding a stale value from a prior
    successful run, so callers can't mistake a failed run's leftover
    state for a real duration.
    """

    def __init__(
    self,
    model: BaseEstimator | None = None,
    ):
        if model is None:
            model = ModelFactory.create()
            
        if not hasattr(model, "fit") or not callable(getattr(model, "fit")):
            
            raise ModelTrainingError(
                f"model must implement a callable fit() method; "
                f"got {type(model).__name__}."
            )

        self.model = model
        self.training_time = None

    @staticmethod
    def _validate_data(
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """
        Validate training inputs.

        Checks emptiness, row-count match, and index alignment between
        X_train and y_train. Index alignment matters because
        `estimator.fit(X, y)` pairs rows positionally, not by label —
        mismatched indices (e.g. from independently filtered frames)
        fit silently on misaligned data without either check_is_fitted
        or sklearn raising any error.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise ModelTrainingError(
                f"X_train must be a pandas DataFrame, got {type(X_train).__name__}."
            )
        if not isinstance(y_train, pd.Series):
            raise ModelTrainingError(
                f"y_train must be a pandas Series, got {type(y_train).__name__}."
            )
        if X_train.empty:
            raise ModelTrainingError("X_train is empty.")
        if y_train.empty:
            raise ModelTrainingError("y_train is empty.")
        if len(X_train) != len(y_train):
            raise ModelTrainingError(
                f"X_train and y_train must have the same number of rows "
                f"(got {len(X_train)} and {len(y_train)})."
            )
        if not X_train.index.equals(y_train.index):
            raise ModelTrainingError(
                "X_train and y_train indices do not match. fit() pairs "
                "rows positionally, so misaligned indices silently "
                "corrupt training. Reset or align indices before training "
                "(e.g. X_train.reset_index(drop=True), "
                "y_train.reset_index(drop=True))."
            )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **fit_kwargs: Any,
    ) -> BaseEstimator:
        """
        Train the model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training labels.
        **fit_kwargs
            Additional arguments passed to fit().

        Returns
        -------
        BaseEstimator
            Trained estimator.

        Raises
        ------
        ModelTrainingError
            If validation fails, or if the underlying fit() call raises.
            On failure, `self.training_time` is reset to None so a
            failed retrain can't be mistaken for a successful one by
            callers reading `get_training_time()` afterward.
        """
        self._validate_data(X_train, y_train)

        self.training_time = None
        start_time = time.perf_counter()
        logger.info(
            "Training %s on %d rows, %d features.",
            type(self.model).__name__,
            len(X_train),
            X_train.shape[1],
        )
        try:
            self.model.fit(
                X_train,
                y_train,
                **fit_kwargs,
            )
        except Exception as exc:
            logger.error("Training failed for %s: %s", type(self.model).__name__, exc)
            raise ModelTrainingError(
                f"Model training failed: {exc}"
            ) from exc

        self.training_time = time.perf_counter() - start_time
        logger.info(
            "Training complete for %s in %.3fs.",
            type(self.model).__name__,
            self.training_time,
        )
        return self.model

    @property
    def is_trained(self) -> bool:
        """
        Returns True if the estimator has already been fitted.
        """
        try:
            check_is_fitted(self.model)
            return True
        except Exception:
            return False

    def get_training_time(self) -> float | None:
        """
        Return training time in seconds for the most recent successful
        `train()` call, or None if the model has not been successfully
        trained (including if the most recent attempt failed).
        """
        return self.training_time