"""
Athena v1.2
--------------------
Feature Registry
Responsibilities
----------------
1. Register all feature generators
2. Execute them in the correct order
3. Provide a single interface for feature engineering
Author: Athena
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

try:
    # Package-relative imports: use these when feature_registry is imported
    # as part of the athena.features package.
    from .amount_features import AmountFeatures
    from .balance_features import BalanceFeatures
    from .behavior_features import BehaviorFeatures
    from .time_features import TimeFeatures
    from .risk_features import RiskFeatures
except ImportError:
    # Fallback for running this file directly or from a flat layout where
    # the *_features.py modules sit next to this file (no package/__init__.py).
    from amount_features import AmountFeatures
    from balance_features import BalanceFeatures
    from behavior_features import BehaviorFeatures
    from time_features import TimeFeatures
    from risk_features import RiskFeatures

# Columns that RiskFeatures expects to already exist on the frame, and the
# upstream generator responsible for producing each. Used as a cheap
# pre-flight check so a broken pipeline order fails at registry.transform()
# with a clear message, instead of RiskFeatures silently zeroing a signal.
_RISK_DEPENDENCY_COLUMNS = {
    "rapid_sender_activity": "BehaviorFeatures",
    "is_first_sender_transaction": "BehaviorFeatures",
    "sender_balance_mismatch": "BalanceFeatures",
    "remaining_sender_ratio": "BalanceFeatures",
}


class FeatureRegistryError(RuntimeError):
    """Raised when the registry's pipeline is misconfigured or a stage fails."""


class FeatureRegistry:
    """
    Central registry for all feature engineering modules.
    New feature generators should be added only here.

    Notes
    -----
    RiskFeatures depends on a fitted large-transaction threshold to avoid
    batch-dependent leakage (see RiskFeatures.fit_threshold). Call
    `fit(reference_df)` once on a training reference set before the first
    `transform()` call; `transform()` will raise if RiskFeatures hasn't
    been fitted and no threshold was supplied at construction time.
    """

    def __init__(
    self,
    risk_large_txn_threshold: Optional[float] = None,
    amount_high_value_threshold: Optional[float] = None,
) -> None:
      self._risk_features = RiskFeatures(
        large_txn_threshold=risk_large_txn_threshold
    )

      self._amount_features = AmountFeatures(
        high_value_threshold=amount_high_value_threshold
    )

      self.feature_generators = [
        self._amount_features,
        BalanceFeatures(),
        BehaviorFeatures(),
        TimeFeatures(),
        self._risk_features,
    ] 

    def fit(
    self,
    reference_df: pd.DataFrame,
    quantile: float = 0.95,
    ) -> "FeatureRegistry":
      """
     Fit all feature generators that require training statistics.
     """

      self._amount_features.fit(reference_df)

      self._risk_features.fit_threshold(
        reference_df,
        quantile=quantile,
    )

      return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute all registered feature generators in order.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame with engineered features.

        Raises
        ------
        FeatureRegistryError
            If a stage fails, or if RiskFeatures would run without a fitted
            threshold and without required upstream columns present.
        """
        if self._risk_features.large_txn_threshold is None:
            raise FeatureRegistryError(
                "RiskFeatures has no fitted large-transaction threshold. "
                "Call registry.fit(reference_df) on a training reference set "
                "before transform(), or construct FeatureRegistry(risk_large_txn_threshold=...)."
            )

        # Single copy up front; individual generators no longer need to
        # each defensively copy the frame themselves.
        data = df.copy()

        for generator in self.feature_generators:
            stage_name = generator.__class__.__name__
            try:
                data = generator.transform(data)
            except Exception as exc:
                raise FeatureRegistryError(
                    f"Feature generator '{stage_name}' failed: {exc}"
                ) from exc

        missing = [c for c in _RISK_DEPENDENCY_COLUMNS if c not in data.columns]
        if missing:
            culprits = sorted({_RISK_DEPENDENCY_COLUMNS[c] for c in missing})
            raise FeatureRegistryError(
                f"Pipeline completed but expected risk-dependency column(s) "
                f"{missing} are missing (expected from {culprits}). "
                f"Check generator ordering or upstream schema changes."
            )

        return data

    def register(self, feature_generator, before_risk: bool = True) -> None:
        """
        Register a new feature generator.

        Parameters
        ----------
        feature_generator
            An object exposing `.transform(df) -> df`.
        before_risk : bool, default True
            If True (default), inserts the generator immediately before
            RiskFeatures, preserving RiskFeatures as the final stage since
            it depends on upstream columns. If False, appends at the very
            end (only safe if the new generator has no dependents).

        Example
        -------
        registry.register(GraphFeatures())
        """
        if before_risk:
            risk_index = self.feature_generators.index(self._risk_features)
            self.feature_generators.insert(risk_index, feature_generator)
        else:
            self.feature_generators.append(feature_generator)

    def list_features(self) -> List[str]:
        """
        Return the names of all registered generators, in execution order.
        """
        return [
            generator.__class__.__name__
            for generator in self.feature_generators
        ]
    