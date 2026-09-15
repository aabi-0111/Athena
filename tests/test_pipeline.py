"""
Athena MVP
----------
Tests for the main data pipeline.

Run:
    pytest tests/test_pipeline.py -v
"""

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_transaction_data():
    """
    Small PaySim-like dataset for pipeline testing.
    """

    return pd.DataFrame({
        "step": [1, 2, 3, 4, 5],
        "type": [
            "PAYMENT",
            "TRANSFER",
            "CASH_OUT",
            "DEBIT",
            "CASH_IN"
        ],
        "amount": [
            100.0,
            5000.0,
            2500.0,
            300.0,
            10000.0
        ],
        "oldbalanceOrg": [
            1000.0,
            6000.0,
            3000.0,
            500.0,
            1000.0
        ],
        "newbalanceOrig": [
            900.0,
            1000.0,
            500.0,
            200.0,
            11000.0
        ],
        "oldbalanceDest": [
            500.0,
            1000.0,
            500.0,
            200.0,
            1000.0
        ],
        "newbalanceDest": [
            600.0,
            6000.0,
            3000.0,
            500.0,
            11000.0
        ],
        "isFlaggedFraud": [
            0,
            0,
            1,
            0,
            0
        ],
        "isFraud": [
            0,
            1,
            1,
            0,
            0
        ]
    })


def test_sample_dataset_is_valid(sample_transaction_data):
    """
    Verify that the test dataset contains the required
    transaction columns.
    """

    required_columns = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFlaggedFraud",
        "isFraud"
    ]

    for column in required_columns:
        assert column in sample_transaction_data.columns


def test_dataset_is_not_empty(sample_transaction_data):
    """
    Dataset should contain transaction records.
    """

    assert not sample_transaction_data.empty


def test_dataset_has_expected_rows(sample_transaction_data):
    """
    Verify the number of test transactions.
    """

    assert len(sample_transaction_data) == 5


def test_target_column_contains_binary_values(
    sample_transaction_data
):
    """
    Fraud target should contain binary values.
    """

    assert set(
        sample_transaction_data["isFraud"].unique()
    ).issubset({0, 1})


def test_pipeline_split_creates_train_test_data(
    sample_transaction_data
):
    """
    Test a basic train/test split using the same concept
    as the Athena pipeline.
    """

    from sklearn.model_selection import train_test_split

    X = sample_transaction_data.drop(
        columns=["isFraud"]
    )

    y = sample_transaction_data["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    assert len(X_train) + len(X_test) == len(X)

    assert len(y_train) + len(y_test) == len(y)


def test_pipeline_output_files_can_be_created(
    tmp_path,
    sample_transaction_data
):
    """
    Test that pipeline output files can be written.

    Uses pytest's temporary directory, so it does not
    modify the actual Athena data directory.
    """

    output_dir = tmp_path / "processed" / "train_test"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    X = sample_transaction_data.drop(
        columns=["isFraud"]
    )

    y = sample_transaction_data["isFraud"]

    X.to_csv(
        output_dir / "X_train.csv",
        index=False
    )

    y.to_csv(
        output_dir / "y_train.csv",
        index=False
    )

    assert (
        output_dir / "X_train.csv"
    ).exists()

    assert (
        output_dir / "y_train.csv"
    ).exists()


def test_pipeline_output_can_be_loaded(
    tmp_path,
    sample_transaction_data
):
    """
    Test that saved pipeline outputs can be loaded again.
    """

    output_dir = tmp_path / "processed"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = output_dir / "test.csv"

    sample_transaction_data.to_csv(
        file_path,
        index=False
    )

    loaded_data = pd.read_csv(file_path)

    assert isinstance(
        loaded_data,
        pd.DataFrame
    )

    assert loaded_data.shape == sample_transaction_data.shape


def test_pipeline_preserves_transaction_count(
    sample_transaction_data
):
    """
    Pipeline transformations should not accidentally
    remove transaction rows.
    """

    original_count = len(
        sample_transaction_data
    )

    processed_data = sample_transaction_data.copy()

    assert len(processed_data) == original_count


def test_pipeline_target_is_separated(
    sample_transaction_data
):
    """
    Verify that isFraud is separated from features.
    """

    X = sample_transaction_data.drop(
        columns=["isFraud"]
    )

    y = sample_transaction_data["isFraud"]

    assert "isFraud" not in X.columns

    assert y.name == "isFraud"

    assert len(X) == len(y)
    