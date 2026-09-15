"""
Athena v1.1
--------------------
Main Pipeline

Responsibilities
----------------
1. Load raw dataset
2. Preprocess dataset
3. Split before any fit-dependent feature engineering
4. Fit feature engineering on training data only
5. Transform training and test data with the fitted transformer
6. Persist cleaned, engineered, and split datasets
"""

from __future__ import annotations

from app.core.constants import CLEANED_DATA_PATH, ENGINEERED_DATA_PATH
from app.core.logger import get_logger
from app.pipeline.data_loader import DataLoader
from app.pipeline.preprocess import DataPreprocessor
from app.pipeline.split_data import DataSplitter
from app.ml.features.feature_engineering import FeatureEngineering

logger = get_logger(__name__)


class AthenaPipeline:
    """Main end-to-end Athena data pipeline."""

    def run(self):
        logger.info("Starting Athena Pipeline...")

        # 1. Load raw dataset
        df = DataLoader().load()

        # 2. Preprocess dataset
        cleaned_df = DataPreprocessor(df).process()
        CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
        logger.info("Clean dataset saved: %s", CLEANED_DATA_PATH)

        # 3. Split BEFORE feature-engineering fit.
        # Any learned feature state (e.g. amount/risk quantiles) must only
        # see the training partition to prevent test-set leakage.
        splitter = DataSplitter()
        X_train_raw, X_test_raw, y_train, y_test = splitter.split(cleaned_df)

        train_df = X_train_raw.copy()
        train_df[splitter.target_column] = y_train
        test_df = X_test_raw.copy()
        test_df[splitter.target_column] = y_test

        # 4. Fit feature engineering exclusively on train data.
        engineer = FeatureEngineering()
        engineer.fit(train_df)

        # 5. Transform both partitions using the train-fitted state.
        train_engineered = engineer.transform(train_df)
        test_engineered = engineer.transform(test_df)

        # Preserve a combined engineered artifact for inspection while keeping
        # the actual model split isolated and already persisted by DataSplitter.
        engineered_df = __import__("pandas").concat(
            [train_engineered, test_engineered],
            axis=0,
            ignore_index=True,
        )
        ENGINEERED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        engineered_df.to_csv(ENGINEERED_DATA_PATH, index=False)
        logger.info("Engineered dataset saved: %s", ENGINEERED_DATA_PATH)

        # Return model-ready partitions without re-fitting feature state.
        X_train = train_engineered.drop(columns=[splitter.target_column, "nameOrig", "nameDest"], errors="ignore")
        X_test = test_engineered.drop(columns=[splitter.target_column, "nameOrig", "nameDest"], errors="ignore")

        logger.info("Pipeline completed successfully.")
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    AthenaPipeline().run()
