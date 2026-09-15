"""
Athena v1.0
--------------------
Hyperparameter Tuning

Responsibilities
----------------
1. Define per-model Optuna search spaces.
2. Run stratified k-fold cross-validated trials.
3. Optimize for PR-AUC (not accuracy), matching the evaluation
   priorities in PRD §15 for a severely imbalanced fraud target.
4. Return the best hyperparameters found, ready to pass into
   ModelFactory.create(model_type, **best_params).

Author: Athena
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold

try:
    from app.core.exceptions import ModelTrainingError
except ImportError:
    from core.exceptions import ModelTrainingError

from app.ml.models.model_factory import ModelFactory, ModelType
from app.ml.evaluation.metrics import MetricsCalculator, MetricsError

__all__ = [
    "HyperparameterTuner",
    "HyperparameterTuningError",
    "TuningResult",
]

logger = logging.getLogger(__name__)

# Silence Optuna's per-trial INFO spam; trial-level results are logged
# explicitly by this module instead, at a level under the caller's control.
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterTuningError(ModelTrainingError):
    """Raised when a tuning study cannot be run or produces no valid trial."""


@dataclass(frozen=True)
class TuningResult:
    """
    Outcome of a completed tuning study.

    Attributes
    ----------
    best_params : dict
        Best hyperparameters found, directly usable as
        ModelFactory.create(model_type, **best_params).
    best_score : float
        Mean cross-validated PR-AUC of the best trial.
    n_trials : int
        Number of trials actually completed (may be less than requested
        if some trials were pruned/failed and Optuna's own bookkeeping
        still counts them).
    study : optuna.Study
        The underlying Optuna study, for further inspection (e.g.
        optuna.visualization, trial history) or MLflow logging.
    """

    best_params: dict[str, Any]
    best_score: float
    n_trials: int
    study: optuna.Study


# Search-space definitions live here, one function per model type, so
# adding a new tunable model means adding one function + one dict entry
# below rather than touching the tuning loop itself.

def _lightgbm_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _xgboost_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _random_forest_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


_SEARCH_SPACES: dict[ModelType, Callable[[optuna.Trial], dict[str, Any]]] = {
    ModelType.LIGHTGBM: _lightgbm_search_space,
    ModelType.XGBOOST: _xgboost_search_space,
    ModelType.RANDOM_FOREST: _random_forest_search_space,
}


class HyperparameterTuner:
    """
    Runs an Optuna study to find hyperparameters maximizing mean
    cross-validated PR-AUC for a given model type.

    Parameters
    ----------
    model_type : str | ModelType
        Any model registered in ModelFactory that also has a search
        space defined in `_SEARCH_SPACES`.
    n_splits : int, default 5
        Number of stratified CV folds per trial.
    random_state : int, default 42
        Seed for both the CV splitter and Optuna's sampler, so a study
        is reproducible given the same data and n_trials.

    Notes
    -----
    Fixed hyperparameters (e.g. class_weight='balanced', random_state)
    are NOT part of the search space -- they're applied consistently by
    ModelFactory's builder defaults, so tuning only varies the
    parameters that actually affect model capacity/regularization.
    """

    def __init__(
        self,
        model_type: str | ModelType,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        try:
            self.model_type = (
                model_type if isinstance(model_type, ModelType) else ModelType(model_type.lower())
            )
        except ValueError as exc:
            raise HyperparameterTuningError(
                f"Unsupported model_type '{model_type}'. "
                f"Available: {', '.join(m.value for m in ModelType)}."
            ) from exc

        if self.model_type not in _SEARCH_SPACES:
            raise HyperparameterTuningError(
                f"No search space defined for '{self.model_type.value}'. "
                f"Tunable models: {[m.value for m in _SEARCH_SPACES]}."
            )

        if n_splits < 2:
            raise HyperparameterTuningError("n_splits must be >= 2.")

        self.n_splits = n_splits
        self.random_state = random_state
        self._search_space_fn = _SEARCH_SPACES[self.model_type]

    def _cv_pr_auc(self, X: pd.DataFrame, y: pd.Series, params: dict[str, Any]) -> float:
        """
        Mean PR-AUC across stratified folds for one hyperparameter set.

        A fold whose validation split ends up single-class (possible on
        a tiny minority class combined with a bad split) contributes no
        score rather than crashing the whole trial; if EVERY fold is
        single-class the trial is failed explicitly rather than silently
        returning a meaningless 0.0 that Optuna would otherwise chase.
        """
        splitter = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        fold_scores: list[float] = []
        for train_idx, val_idx in splitter.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = ModelFactory.create(self.model_type, **params)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            try:
                metrics = MetricsCalculator.compute(
                    y_true=y_val.to_numpy(), y_pred=y_pred, y_prob=y_prob
                )
            except MetricsError as exc:
                logger.warning("Fold skipped (metrics computation failed): %s", exc)
                continue

            if metrics["pr_auc"] is not None:
                fold_scores.append(metrics["pr_auc"])

        if not fold_scores:
            raise HyperparameterTuningError(
                "No fold produced a valid PR-AUC (every fold's validation "
                "split may have been single-class). Increase n_splits or "
                "check class balance."
            )

        return float(np.mean(fold_scores))

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        timeout: float | None = None,
        show_progress_bar: bool = False,
    ) -> TuningResult:
        """
        Run the tuning study.

        Parameters
        ----------
        X, y : training data (reference/training split only -- never
            pass test data here, or the "best" hyperparameters are
            tuned against data the final model will later be scored on).
        n_trials : int
            Number of Optuna trials.
        timeout : float | None
            Optional wall-clock budget in seconds; the study stops
            early (keeping whatever trials completed) if exceeded.
        show_progress_bar : bool
            Optuna's built-in progress bar; off by default since it
            writes directly to stderr and is unwanted in non-interactive
            runs (e.g. CI, scheduled retraining jobs).

        Returns
        -------
        TuningResult

        Raises
        ------
        HyperparameterTuningError
            If inputs are invalid, or no trial completes successfully.
        """
        if not isinstance(X, pd.DataFrame):
            raise HyperparameterTuningError(
                f"X must be a pandas DataFrame, got {type(X).__name__}."
            )
        if not isinstance(y, pd.Series):
            raise HyperparameterTuningError(
                f"y must be a pandas Series, got {type(y).__name__}."
            )
        if X.empty or y.empty:
            raise HyperparameterTuningError("X and y must not be empty.")
        if len(X) != len(y):
            raise HyperparameterTuningError(
                f"X and y must have the same number of rows "
                f"(got {len(X)} and {len(y)})."
            )
        if n_trials < 1:
            raise HyperparameterTuningError("n_trials must be >= 1.")

        def _objective(trial: optuna.Trial) -> float:
            params = self._search_space_fn(trial)
            return self._cv_pr_auc(X, y, params)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        logger.info(
            "Starting Optuna study | model=%s trials=%d cv_folds=%d",
            self.model_type.value, n_trials, self.n_splits,
        )

        study.optimize(
            _objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress_bar,
        )

        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            raise HyperparameterTuningError(
                f"No trial completed successfully out of {len(study.trials)} "
                "attempted. Check the search space and input data."
            )

        logger.info(
            "Study complete | best_pr_auc=%.5f best_params=%s",
            study.best_value, study.best_params,
        )

        return TuningResult(
            best_params=dict(study.best_params),
            best_score=float(study.best_value),
            n_trials=len(completed),
            study=study,
        )
