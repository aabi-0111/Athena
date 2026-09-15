"""
Athena MVP
----------
Tests for feature engineering.

Run:
    py -m pytest tests/test_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from app.ml.features.feature_engineering import FeatureEngineering


@pytest.fixture
def sample_data():
    """
    Small synthetic PaySim-like dataset used for feature tests.
    Includes all columns required by the feature engineering pipeline.
    """

    return pd.DataFrame({
        "step": [1, 10, 20, 30, 40],

        "type": [
            "PAYMENT",
            "TRANSFER",
            "CASH_OUT",
            "DEBIT",
            "CASH_IN",
        ],

        "amount": [
            100.0,
            5000.0,
            2500.0,
            300.0,
            10000.0,
        ],

        # Required by BehaviorFeatures
        "nameOrig": [
            "C10001",
            "C10002",
            "C10003",
            "C10004",
            "C10005",
        ],

        # Required by BehaviorFeatures
        "nameDest": [
            "M10001",
            "C20001",
            "C20002",
            "M20001",
            "C20003",
        ],

        "oldbalanceOrg": [
            1000.0,
            6000.0,
            3000.0,
            500.0,
            1000.0,
        ],

        "newbalanceOrig": [
            900.0,
            1000.0,
            500.0,
            200.0,
            11000.0,
        ],

        "oldbalanceDest": [
            500.0,
            1000.0,
            500.0,
            200.0,
            1000.0,
        ],

        "newbalanceDest": [
            600.0,
            6000.0,
            3000.0,
            500.0,
            11000.0,
        ],

        "isFlaggedFraud": [
            0,
            0,
            1,
            0,
            0,
        ],
    })


@pytest.fixture
def fitted_feature_engineer(sample_data):
    """
    Create and fit FeatureEngineering before transformation.

    The RiskFeatures module requires a fitted
    large-transaction threshold.
    """

    feature_engineer = FeatureEngineering()

    # Fit using the reference/training dataset
    feature_engineer.fit(sample_data)

    return feature_engineer


def test_feature_engineering_initialization():
    """
    Test FeatureEngineering initialization.
    """

    feature_engineer = FeatureEngineering()

    assert feature_engineer is not None


def test_feature_engineering_returns_dataframe(
    sample_data,
    fitted_feature_engineer,
):
    """
    Test that feature engineering returns a DataFrame.
    """

    result = fitted_feature_engineer.transform(sample_data)

    assert isinstance(result, pd.DataFrame)


def test_feature_engineering_preserves_rows(
    sample_data,
    fitted_feature_engineer,
):
    """
    Feature engineering should preserve
    the number of transaction rows.
    """

    result = fitted_feature_engineer.transform(sample_data)

    assert len(result) == len(sample_data)


def test_feature_engineering_creates_new_features(
    sample_data,
    fitted_feature_engineer,
):
    """
    Feature engineering should create additional
    feature columns.
    """

    result = fitted_feature_engineer.transform(sample_data)

    assert len(result.columns) > len(sample_data.columns)


def test_original_columns_are_preserved(
    sample_data,
    fitted_feature_engineer,
):
    """
    Verify that important original transaction
    columns remain available.
    """

    result = fitted_feature_engineer.transform(sample_data)

    required_columns = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFlaggedFraud",
    ]

    for column in required_columns:
        assert column in result.columns


def test_feature_engineering_has_no_nan_values(
    sample_data,
    fitted_feature_engineer,
):
    """
    Feature engineering should not introduce NaN values.
    """

    result = fitted_feature_engineer.transform(sample_data)

    assert not result.isnull().values.any()


def test_amount_feature_is_numeric(
    sample_data,
    fitted_feature_engineer,
):
    """
    Verify that the amount column remains numeric.
    """

    result = fitted_feature_engineer.transform(sample_data)

    assert pd.api.types.is_numeric_dtype(
        result["amount"]
    )


def test_numeric_features_are_numeric(
    sample_data,
    fitted_feature_engineer,
):
    """
    Verify that numeric transaction columns remain numeric.
    """

    result = fitted_feature_engineer.transform(sample_data)

    numeric_columns = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFlaggedFraud",
    ]

    for column in numeric_columns:
        assert pd.api.types.is_numeric_dtype(
            result[column]
        )


def test_feature_engineering_does_not_modify_original_dataframe(
    sample_data,
    fitted_feature_engineer,
):
    """
    Ensure the original input DataFrame is not modified.
    """

    original_data = sample_data.copy()

    fitted_feature_engineer.transform(sample_data)

    pd.testing.assert_frame_equal(
        sample_data,
        original_data,
    )


def test_feature_engineering_is_consistent(
    sample_data,
    fitted_feature_engineer,
):
    """
    Running transformation twice on the same input
    should produce consistent results.
    """

    result_1 = fitted_feature_engineer.transform(
        sample_data
    )

    result_2 = fitted_feature_engineer.transform(
        sample_data
    )

    pd.testing.assert_frame_equal(
        result_1,
        result_2,
    )


def test_engineered_features_are_finite(
    sample_data,
    fitted_feature_engineer,
):
    """
    Ensure numeric engineered features do not contain
    infinite values.
    """

    result = fitted_feature_engineer.transform(
        sample_data
    )

    numeric_data = result.select_dtypes(
        include=[np.number]
    )

    assert np.isfinite(
        numeric_data.to_numpy()
    ).all()
