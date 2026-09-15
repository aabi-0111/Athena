"""
Athena v1.3
--------------------
Feature Engineering Pipeline
Responsibilities
----------------
1. Central interface for feature engineering
2. Fit feature generators that require training statistics
3. Transform datasets using the registered feature pipeline
4. Support both training and inference workflows
Author: Athena
"""
    # Package-relative import: use this when FeatureEngineering is imported
    # as part of the athena package, e.g.
    #   athena/
    #     feature_engineering.py   <- this file
    #     features/
    #       __init__.py
    #       feature_registry.py
from __future__ import annotations

from typing import Optional

import pandas as pd

from .features_registry import (
    FeatureRegistry,
    FeatureRegistryError,
)

__all__ = [
    "FeatureEngineering",
    "FeatureRegistryError",
]


class FeatureEngineering:
    """
    Wrapper around the FeatureRegistry.

    This class provides a clean interface between preprocessing and
    model training/inference.

    Workflow
    --------
    Training:
        fe = FeatureEngineering()
        fe.fit(train_df)
        train_features = fe.transform(train_df)

    Inference:
        test_features = fe.transform(test_df)

    Persisting fitted state across process restarts (e.g. training and
    inference running as separate services):
        params = fe.get_params()               # after fit(), save via MLflow/etc.
        fe = FeatureEngineering(**params)       # reload at inference time, no re-fit needed
    """

    def __init__(
        self,
        risk_large_txn_threshold: Optional[float] = None,
    ) -> None:
        self.registry = FeatureRegistry(
            risk_large_txn_threshold=risk_large_txn_threshold
        )

    def fit(
        self,
        reference_df: pd.DataFrame,
        quantile: float = 0.95,
    ) -> "FeatureEngineering":
        """
        Fit all feature generators that require reference
        statistics from the training dataset.

        Parameters
        ----------
        reference_df : pd.DataFrame
            Training dataset.
        quantile : float
            Quantile used for fitting the large transaction
            threshold used by RiskFeatures.

        Returns
        -------
        FeatureEngineering
        """
        self.registry.fit(reference_df, quantile=quantile)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate engineered features.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        FeatureRegistryError
            If the pipeline has not been fitted (no large-transaction
            threshold available) or if any feature stage fails. This
            is the single source of truth for fitted-state validation;
            the registry itself checks its fitted RiskFeatures instance,
            so this wrapper doesn't duplicate that state separately.
        """
        return self.registry.transform(df)

    def fit_transform(
        self,
        df: pd.DataFrame,
        quantile: float = 0.95,
    ) -> pd.DataFrame:
        """
        Fit the pipeline and transform the same dataset.
        Used during training.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        self.fit(df, quantile=quantile)
        return self.transform(df)

    def get_params(self) -> dict:
        """
        Return fitted pipeline parameters needed to reconstruct this
        object's state elsewhere (e.g. a separate inference process)
        without re-fitting on training data.

        Returns
        -------
        dict
            Suitable for `FeatureEngineering(**params)` or for logging
            to an experiment tracker (e.g. MLflow params).
        """
        return {
            "risk_large_txn_threshold": self.registry._risk_features.large_txn_threshold,
        }

    @property
    def is_fitted(self) -> bool:
        """Whether the pipeline has a usable (fitted or supplied) risk threshold."""
        return self.registry._risk_features.large_txn_threshold is not None

    @property
    def feature_generators(self):
        """
        Returns the registered feature generators, in execution order.
        """
        return self.registry.list_features()
    