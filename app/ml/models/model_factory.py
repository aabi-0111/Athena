"""
Athena v1.0
--------------------
Model Factory

Responsibilities
----------------
1. Provide a centralized factory for creating ML models.
2. Decouple training code from model implementations.
3. Support easy extension with additional algorithms.
"""

from __future__ import annotations

import importlib
from sklearn.base import BaseEstimator
from enum import Enum
from typing import Any, Callable, Dict, Tuple

class ModelFactoryError(ValueError):
    """Raised when an unsupported model is requested, or a model
    cannot be built from the given arguments."""


class ModelType(str, Enum):
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"


class ModelFactory:
    """
    Factory for constructing machine-learning models.

    Builder implementations are resolved lazily (see `_BUILDER_PATHS`)
    rather than imported at module load time. This is what actually
    delivers responsibility #2, "decouple training code from model
    implementations": importing `model_factory` no longer transitively
    imports every registered model module (and their dependencies —
    e.g. `xgboost`) whether or not that model is ever requested. A
    deployment that only serves the random-forest model doesn't need
    xgboost installed at all; it now fails only if XGBOOST is actually
    requested, with a clear message, instead of at import time.

    Examples
    --------
    >>> model = ModelFactory.create("random_forest")
    >>> model = ModelFactory.create(
    ...     "xgboost",
    ...     n_estimators=500,
    ... )
    """

    # (module path, attribute name) — resolved via importlib on first use,
    # not at class-definition time.
     
    _BUILDER_PATHS = {
    ModelType.RANDOM_FOREST: (
        "app.ml.models.random_forest",
        "build_random_forest",
    ),
    ModelType.XGBOOST: (
    "app.ml.models.xgboost_model",
    "build_xgboost_model",
    ),
} 

    # Populated lazily; caches the resolved builder callable per model type
    # so repeated `create()` calls for the same type don't re-impo
    _builder_cache: Dict[ModelType, Callable[..., BaseEstimator]] = {}

    @classmethod
    def available_models(cls) -> tuple[str, ...]:
        """
        Return supported model names.
        """
        return tuple(model.value for model in ModelType)

    @classmethod
    def _resolve_builder(cls, model_enum: ModelType) -> Callable[..., BaseEstimator]:
        """
        Import and cache the builder function for `model_enum`, on first
        use only.
        """
        if model_enum in cls._builder_cache:
            return cls._builder_cache[model_enum]

        module_path, attr_name = cls._BUILDER_PATHS[model_enum]
        try:
            module = importlib.import_module(module_path)
            builder = getattr(module, attr_name)
        except ImportError as exc:
            raise ModelFactoryError(
                f"Model '{model_enum.value}' requires '{module_path}', which "
                f"could not be imported (missing dependency?): {exc}"
            ) from exc
        except AttributeError as exc:
            raise ModelFactoryError(
                f"Model '{model_enum.value}' is registered to "
                f"'{module_path}.{attr_name}', but that function does not exist."
            ) from exc

        cls._builder_cache[model_enum] = builder
        return builder

    @classmethod
    def create(
    cls,
    model_type: str | ModelType = ModelType.XGBOOST,
    **kwargs,
    ) -> BaseEstimator:
        """
        Create a model instance.

        Parameters
        ----------
        model_type : str or ModelType
            Model identifier.
        **kwargs
            Passed directly to the model builder.

        Returns
        -------
        estimator
            Untrained sklearn/xgboost estimator.

        Raises
        ------
        ModelFactoryError
            If the model type is unsupported, its implementation module
            can't be imported, or the builder rejects `kwargs`.
        """
        if not isinstance(model_type, (str, ModelType)):
            raise ModelFactoryError(
                f"model_type must be a str or ModelType, got {type(model_type).__name__}."
            )

        try:
            model_enum = (
                model_type
                if isinstance(model_type, ModelType)
                else ModelType(model_type.lower())
            )
        except ValueError as exc:
            raise ModelFactoryError(
                f"Unsupported model '{model_type}'. "
                f"Available models: {', '.join(cls.available_models())}."
            ) from exc

        builder = cls._resolve_builder(model_enum)

        try:
            return builder(**kwargs)
        except TypeError as exc:
            # Builders are plain functions/constructors — a typo'd
            # hyperparameter (easy to hit when kwargs come from an Optuna
            # trial) surfaces here as a generic TypeError. Re-raising as
            # ModelFactoryError keeps it catchable alongside every other
            # factory failure, with the offending model type attached.
            raise ModelFactoryError(
                f"Failed to build model '{model_enum.value}' with arguments "
                f"{kwargs}: {exc}"
            ) from exc
        