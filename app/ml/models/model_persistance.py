"""
Athena v1.0
--------------------
Model Persistence

Responsibilities
----------------
1. Save trained machine-learning models.
2. Load previously saved models.
3. Validate model objects and filesystem paths.
4. Perform atomic writes to prevent file corruption.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Final

import joblib
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

__all__ = [
    "ModelPersistence",
    "ModelPersistenceError",
]

logger = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".pkl",
        ".joblib",
    }
)


class ModelPersistenceError(RuntimeError):
    """
    Raised when a model cannot be saved or loaded.
    """


class ModelPersistence:
    """
    Save and load trained machine-learning models.

    Notes
    -----
    Models are written atomically.

    The model is first written to a temporary file in the destination
    directory and then moved into place using ``os.replace()``. This
    guarantees that interruptions never leave behind a partially-written
    model file.
    """

    @staticmethod
    def _validate_extension(path: Path) -> None:
        """
        Validate the model file extension.
        """
        if path.suffix.lower() not in _ALLOWED_EXTENSIONS:
            raise ModelPersistenceError(
                f"Unsupported model extension '{path.suffix}'. "
                f"Expected one of {sorted(_ALLOWED_EXTENSIONS)}."
            )

    @staticmethod
    def _validate_model(model: BaseEstimator) -> None:
        """
        Validate that the estimator is fitted.

        Raises
        ------
        ModelPersistenceError
            If the model is None, not an estimator,
            or has not been fitted.
        """
        if model is None:
            raise ModelPersistenceError(
                "Cannot save None as a model."
            )

        if not isinstance(model, BaseEstimator):
            raise ModelPersistenceError(
                f"Expected sklearn BaseEstimator, "
                f"got {type(model).__name__}."
            )

        try:
            check_is_fitted(model)
        except Exception as exc:
            raise ModelPersistenceError(
                "Cannot save an unfitted model. "
                "Train the model before saving."
            ) from exc

    @classmethod
    def save_model(
        cls,
        model: BaseEstimator,
        output_path: str | Path,
    ) -> Path:
        """
        Save a trained model.

        Parameters
        ----------
        model : BaseEstimator
            Trained estimator.

        output_path : str | Path
            Destination path.

        Returns
        -------
        Path
            Saved model path.

        Raises
        ------
        ModelPersistenceError
            If validation or saving fails.
        """
        cls._validate_model(model)

        output_path = Path(output_path).expanduser().resolve()

        cls._validate_extension(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Saving trained model to %s",
            output_path,
        )

        fd, tmp_name = tempfile.mkstemp(
            dir=output_path.parent,
            prefix=".tmp-",
            suffix=output_path.suffix,
        )

        os.close(fd)

        tmp_path = Path(tmp_name)

        try:

            joblib.dump(
                model,
                tmp_path,
            )

            os.replace(
                tmp_path,
                output_path,
            )

        except Exception as exc:

            tmp_path.unlink(
                missing_ok=True,
            )

            logger.exception(
                "Failed to save model."
            )

            raise ModelPersistenceError(
                f"Failed to save model to "
                f"'{output_path}': {exc}"
            ) from exc

        logger.info(
            "Model successfully saved."
        )

        return output_path

    @classmethod
    def load_model(
        cls,
        model_path: str | Path,
    ) -> BaseEstimator:
        """
        Load a trained model.

        Parameters
        ----------
        model_path : str | Path

        Returns
        -------
        BaseEstimator
            Loaded estimator.

        Raises
        ------
        ModelPersistenceError
            If loading or validation fails.
        """
        model_path = Path(
            model_path
        ).expanduser().resolve()

        cls._validate_extension(
            model_path,
        )

        if not model_path.exists():
            raise ModelPersistenceError(
                f"Model file does not exist: "
                f"{model_path}"
            )

        if not model_path.is_file():
            raise ModelPersistenceError(
                f"Expected a file, got: "
                f"{model_path}"
            )

        logger.info(
            "Loading model from %s",
            model_path,
        )

        try:

            model = joblib.load(
                model_path,
            )

        except Exception as exc:

            logger.exception(
                "Failed to load model."
            )

            raise ModelPersistenceError(
                f"Failed to load model "
                f"'{model_path}': {exc}"
            ) from exc

        if not isinstance(
            model,
            BaseEstimator,
        ):
            raise ModelPersistenceError(
                f"Loaded object is not a sklearn estimator "
                f"(got {type(model).__name__})."
            )

        try:

            check_is_fitted(
                model,
            )

        except Exception as exc:

            raise ModelPersistenceError(
                "Loaded model is not fitted."
            ) from exc

        logger.info(
            "Model loaded successfully."
        )

        return model
    