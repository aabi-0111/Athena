"""
Athena v1.0
--------------------
Model Tests

Tests
----------------
1. Random Forest model creation.
2. XGBoost model creation.
3. Model training.
4. Prediction output.
5. Probability output.
6. ModelFactory creation.
7. Invalid model handling.
8. ModelTrainer initialization.

Run:
    py -m pytest tests/test_models.py -v
"""

import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from app.ml.models.random_forest import build_random_forest
from app.ml.models.xgboost_model import build_xgboost_model
from app.ml.models.model_factory import (
    ModelFactory,
    ModelFactoryError,
)
from app.ml.models.train import ModelTrainer


# ---------------------------------------------------------------------------
# Test Dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_training_data():
    """
    Create a small synthetic binary classification dataset.

    The dataset is intentionally small so model tests execute quickly.
    """

    X = pd.DataFrame({
        "feature_1": [
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
            0.9, 1.0, 1.1, 1.2,
        ],
        "feature_2": [
            1.0, 1.1, 1.2, 1.3,
            2.0, 2.1, 2.2, 2.3,
            3.0, 3.1, 3.2, 3.3,
        ],
        "feature_3": [
            10, 20, 30, 40,
            50, 60, 70, 80,
            90, 100, 110, 120,
        ],
    })

    y = pd.Series(
        [
            0, 0, 0, 0,
            1, 1, 1, 1,
            0, 1, 0, 1,
        ],
        name="isFraud",
    )

    return X, y


# ---------------------------------------------------------------------------
# Model Creation Tests
# ---------------------------------------------------------------------------

def test_random_forest_creation():
    """
    Test that Random Forest builder returns a valid classifier.
    """

    model = build_random_forest(
        n_estimators=10,
        random_state=42,
    )

    assert model is not None
    assert isinstance(model, RandomForestClassifier)


def test_xgboost_creation():
    """
    Test that XGBoost builder returns a valid classifier.
    """

    model = build_xgboost_model(
        n_estimators=10,
        random_state=42,
    )

    assert model is not None
    assert isinstance(model, XGBClassifier)


# ---------------------------------------------------------------------------
# Model Training Tests
# ---------------------------------------------------------------------------

def test_random_forest_training(sample_training_data):
    """
    Test that Random Forest can be trained successfully.
    """

    X, y = sample_training_data

    model = build_random_forest(
        n_estimators=10,
        random_state=42,
    )

    model.fit(X, y)

    assert hasattr(model, "classes_")
    assert hasattr(model, "n_features_in_")
    assert model.n_features_in_ == X.shape[1]


def test_xgboost_training(sample_training_data):
    """
    Test that XGBoost can be trained successfully.
    """

    X, y = sample_training_data

    model = build_xgboost_model(
        n_estimators=10,
        random_state=42,
    )

    model.fit(X, y)

    assert hasattr(model, "classes_")
    assert hasattr(model, "n_features_in_")
    assert model.n_features_in_ == X.shape[1]


# ---------------------------------------------------------------------------
# Prediction Tests
# ---------------------------------------------------------------------------

def test_random_forest_prediction_values(sample_training_data):
    """
    Test Random Forest prediction output.
    """

    X, y = sample_training_data

    model = build_random_forest(
        n_estimators=10,
        random_state=42,
    )

    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})


def test_xgboost_prediction_values(sample_training_data):
    """
    Test XGBoost prediction output.
    """

    X, y = sample_training_data

    model = build_xgboost_model(
        n_estimators=10,
        random_state=42,
    )

    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})


# ---------------------------------------------------------------------------
# Probability Tests
# ---------------------------------------------------------------------------

def test_random_forest_probability_output(sample_training_data):
    """
    Test Random Forest probability predictions.
    """

    X, y = sample_training_data

    model = build_random_forest(
        n_estimators=10,
        random_state=42,
    )

    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(X), 2)
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)

    # Each row's class probabilities must sum to 1.
    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    )


def test_xgboost_probability_output(sample_training_data):
    """
    Test XGBoost probability predictions.
    """

    X, y = sample_training_data

    model = build_xgboost_model(
        n_estimators=10,
        random_state=42,
    )

    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(X), 2)
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)

    # Each row's class probabilities must sum to 1.
    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    )


# ---------------------------------------------------------------------------
# ModelFactory Tests
# ---------------------------------------------------------------------------

def test_model_factory_random_forest():
    """
    Test ModelFactory creation of Random Forest.
    """

    model = ModelFactory.create(
        "random_forest",
        n_estimators=10,
        random_state=42,
    )

    assert model is not None
    assert isinstance(model, RandomForestClassifier)


def test_model_factory_xgboost():
    """
    Test ModelFactory creation of XGBoost.
    """

    model = ModelFactory.create(
        "xgboost",
        n_estimators=10,
        random_state=42,
    )

    assert model is not None
    assert isinstance(model, XGBClassifier)


def test_model_factory_invalid_model():
    """
    Test that an unsupported model type raises ModelFactoryError.
    """

    with pytest.raises(ModelFactoryError):
        ModelFactory.create("invalid_model")


def test_model_factory_available_models():
    """
    Test that ModelFactory reports supported model types.
    """

    available_models = ModelFactory.available_models()

    assert "random_forest" in available_models
    assert "xgboost" in available_models


# ---------------------------------------------------------------------------
# ModelTrainer Tests
# ---------------------------------------------------------------------------

def test_model_trainer_initialization():
    """
    Test ModelTrainer can be initialized.
    """

    trainer = ModelTrainer()

    assert trainer is not None