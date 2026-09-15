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
6. Persist cleaned and engineered datasets
"""

from __future__ import annotations

import pandas as pd

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

        df = DataLoader().load()

        cleaned_df = DataPreprocessor(df).process()
        CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
        logger.info("Clean dataset saved: %s", CLEANED_DATA_PATH)

        # Split before fitting any feature transformer whose state depends on
        # the data distribution (for example, the high-amount quantile).
        splitter = DataSplitter(drop_identifier_columns=False)
        X_train_raw, X_test_raw, y_train, y_test = splitter.split(cleaned_df)

        train_df = X_train_raw.copy()
        train_df[splitter.target_column] = y_train
        test_df = X_test_raw.copy()
        test_df[splitter.target_column] = y_test

        # Fit ONLY on training data; transform both partitions with that state.
        engineer = FeatureEngineering()
        engineer.fit(train_df)
        train_engineered = engineer.transform(train_df)
        test_engineered = engineer.transform(test_df)

        engineered_df = pd.concat(
            [train_engineered, test_engineered],
            axis=0,
            ignore_index=True,
        )
        ENGINEERED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        engineered_df.to_csv(ENGINEERED_DATA_PATH, index=False)
        logger.info("Engineered dataset saved: %s", ENGINEERED_DATA_PATH)

        drop_model_columns = [splitter.target_column, "nameOrig", "nameDest"]
        X_train = train_engineered.drop(columns=drop_model_columns, errors="ignore")
        X_test = test_engineered.drop(columns=drop_model_columns, errors="ignore")

        logger.info("Pipeline completed successfully.")
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    AthenaPipeline().run()
