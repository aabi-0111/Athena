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
from typing import Any, Callable, Dict

from app.core.exceptions import ModelFactoryError


class ModelType(str, Enum):
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"


class ModelFactory:
    """
    Factory for constructing machine-learning models.

    Builder implementations are resolved lazily (see `_BUILDER_PATHS`)
    rather than imported at module load time.
    """

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

    _builder_cache: Dict[ModelType, Callable[..., BaseEstimator]] = {}

    @classmethod
    def available_models(cls) -> tuple[str, ...]:
        return tuple(model.value for model in ModelType)

    @classmethod
    def _resolve_builder(cls, model_enum: ModelType) -> Callable[..., BaseEstimator]:
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
        **kwargs: Any,
    ) -> BaseEstimator:
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
            raise ModelFactoryError(
                f"Failed to build model '{model_enum.value}' with arguments "
                f"{kwargs}: {exc}"
            ) from exc
