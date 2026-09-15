"""
Athena v1.1
--------------------
Main Pipeline

Responsibilities
----------------
1. Load raw dataset
2. Preprocess dataset
3. Save cleaned dataset
4. Run feature engineering
5. Save engineered dataset
6. Perform train/test split
7. Save split datasets

Author: Athena
"""

from pathlib import Path

from app.core.logger import get_logger
from app.pipeline.data_loader import DataLoader
from app.pipeline.preprocess import DataPreprocessor
from app.pipeline.split_data import DataSplitter
from app.ml.features.feature_engineering import FeatureEngineering


logger = get_logger(__name__)


class AthenaPipeline:
    """Main end-to-end Athena data pipeline."""

    OUTPUT_DIR = Path("data") / "processed"

    def __init__(self):
        self.OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

    def run(self):
        logger.info(
            "Starting Athena Pipeline..."
        )

        # ==========================================================
        # 1. Load raw dataset
        # ==========================================================

        loader = DataLoader()

        df = loader.load()

        # ==========================================================
        # 2. Preprocess dataset
        # ==========================================================

        preprocessor = DataPreprocessor(df)

        cleaned_df = preprocessor.process()

        cleaned_path = (
            self.OUTPUT_DIR /
            "cleaned.csv"
        )

        cleaned_df.to_csv(
            cleaned_path,
            index=False
        )

        logger.info(
            "Clean dataset saved: %s",
            cleaned_path
        )

        # ==========================================================
        # 3. Feature Engineering
        # ==========================================================

        engineer = FeatureEngineering()

        engineered_df = engineer.fit_transform(
            cleaned_df
        )

        engineered_path = (
            self.OUTPUT_DIR /
            "engineered.csv"
        )

        engineered_df.to_csv(
            engineered_path,
            index=False
        )

        logger.info(
            "Engineered dataset saved: %s",
            engineered_path
        )

        # ==========================================================
        # 4. Train/Test Split
        # ==========================================================

        splitter = DataSplitter()

        X_train, X_test, y_train, y_test = splitter.split(
            engineered_df
        )

        logger.info(
            "Train/Test split completed successfully."
        )

        # ==========================================================
        # Pipeline Finished
        # ==========================================================

        logger.info(
            "Pipeline completed successfully."
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test,
        )


if __name__ == "__main__":
    pipeline = AthenaPipeline()
    pipeline.run()