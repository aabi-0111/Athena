"""
Athena v1.0
--------------------
Custom Exception Classes

Responsibilities
----------------
1. Define all project-specific exceptions.
2. Provide meaningful error messages.
3. Allow centralized exception handling.
"""


class AthenaError(Exception):
    """
    Base exception for all Athena errors.
    Every custom exception should inherit from this class.
    """

    def __init__(self, message: str = "An unknown Athena error occurred."):
        super().__init__(message)
        self.message = message


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

class ConfigurationError(AthenaError):
    """Raised when configuration values are invalid or missing."""
    pass


# ---------------------------------------------------------------------
# Data Errors
# ---------------------------------------------------------------------

class DataError(AthenaError):
    """Base class for all data-related errors."""
    pass


class DataLoadError(DataError):
    """Raised when a dataset cannot be loaded."""
    pass


class DataValidationError(DataError):
    """Raised when dataset schema validation fails."""
    pass


class MissingColumnError(DataValidationError):
    """Raised when required columns are missing."""
    pass


class InvalidDataTypeError(DataValidationError):
    """Raised when a column has an unexpected datatype."""
    pass

class SchemaValidationError(DataValidationError):
    """
    Backward-compatible alias for schema validation failures.
    """
    pass


class DataIntegrityError(DataError):
    """
    Raised when data integrity checks fail.
    """
    pass

class EmptyDatasetError(DataError):
    """Raised when the dataset is empty."""
    pass


class PreprocessingError(DataError):
    """Raised when preprocessing fails."""
    pass


class FeatureEngineeringError(DataError):
    """Raised when feature engineering fails."""
    pass


class DataSplitError(DataError):
    """Raised when train-test split fails."""
    pass


# ---------------------------------------------------------------------
# Model Errors
# ---------------------------------------------------------------------

class ModelError(AthenaError):
    """Base class for model-related exceptions."""
    pass


class ModelTrainingError(ModelError):
    """Raised when model training fails."""
    pass


class ModelLoadingError(ModelError):
    """Raised when loading a saved model fails."""
    pass


class ModelSavingError(ModelError):
    """Raised when saving a model fails."""
    pass


class PredictionError(ModelError):
    """Raised during prediction failures."""
    pass


class EvaluationError(ModelError):
    """Raised during evaluation failures."""
    pass


# ---------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------

class ExplainabilityError(AthenaError):
    """Raised when SHAP/XAI computation fails."""
    pass


# ---------------------------------------------------------------------
# API Errors
# ---------------------------------------------------------------------

class APIError(AthenaError):
    """Base API exception."""
    pass


class RequestValidationError(APIError):
    """Raised when an API request is invalid."""
    pass


# ---------------------------------------------------------------------
# File Errors
# ---------------------------------------------------------------------

class FileOperationError(AthenaError):
    """Raised when file read/write operations fail."""
    pass


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def raise_if(condition: bool, exception_cls, message: str):
    """
    Raise an exception if a condition is True.

    Example
    -------
    raise_if(df.empty, EmptyDatasetError, "Dataset is empty.")
    """
    if condition:
        raise exception_cls(message)
    