"""
Athena v1.0
--------------------
Training Pipeline

Responsibilities
----------------
1. Load train/test datasets.
2. Train ML model.
3. Save trained model.
4. Evaluate model.
5. Save evaluation artifacts.
6. Generate SHAP explanations.

Author: Athena
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.core.logger import get_logger
from app.ml.models.train import ModelTrainer
from app.ml.models.model_persistance import ModelPersistence
from app.ml.evaluation.evaluate import ModelEvaluator
from app.ml.evaluation.confusion_metrix import ConfusionMatrixGenerator
from app.ml.evaluation.feature_importance import FeatureImportanceExtractor
from app.ml.explainability.shap_explainer import SHAPExplainer
from app.ml.models.model_factory import ModelFactory

trainer = ModelTrainer(

    ModelFactory.create()

)
logger = get_logger(__name__)


class TrainingPipeline:
    """
    End-to-end ML training pipeline.
    """

    DATA_DIR = Path("data") / "processed" / "train_test"

    MODEL_DIR = Path("saved_models")

    REPORT_DIR = Path("reports")

    def __init__(self):

        self.MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ----------------------------------------------------------

    def _load_split_data(self):

        logger.info("Loading train/test datasets...")

        X_train = pd.read_csv(self.DATA_DIR / "X_train.csv")

        X_test = pd.read_csv(self.DATA_DIR / "X_test.csv")

        y_train = pd.read_csv(self.DATA_DIR / "y_train.csv").squeeze()

        y_test = pd.read_csv(self.DATA_DIR / "y_test.csv").squeeze()

        logger.info("Train/Test datasets loaded successfully.")

        return X_train, X_test, y_train, y_test

    # ----------------------------------------------------------

    def run(self):

        logger.info("=" * 60)
        logger.info("Starting Athena Training Pipeline")
        logger.info("=" * 60)

        # ------------------------------------------------------
        # Load Data
        # ------------------------------------------------------

        X_train, X_test, y_train, y_test = self._load_split_data()

        # ------------------------------------------------------
        # Train Model
        # ------------------------------------------------------

        trainer = ModelTrainer()

        model = trainer.train(
            X_train,
            y_train,
        )

        logger.info("Model training completed.")

        # ------------------------------------------------------
        # Save Model
        # ------------------------------------------------------

        ModelPersistence.save_model(
            model,
            self.MODEL_DIR / "best_model.pkl",
        )

        logger.info("Model saved successfully.")

        # ------------------------------------------------------
        # Evaluate
        # ------------------------------------------------------

        results = ModelEvaluator.evaluate(
            model=model,
            X_test=X_test,
            y_test=y_test,
        )

        logger.info("Evaluation completed.")

        # ------------------------------------------------------
        # Save Feature Importance
        # ------------------------------------------------------

        if results["feature_importance"] is not None:

            FeatureImportanceExtractor.save_csv(
                results["feature_importance"],
                self.REPORT_DIR / "feature_importance.csv",
            )

        # ------------------------------------------------------
        # Save Confusion Matrix
        # ------------------------------------------------------

        ConfusionMatrixGenerator.save_plot(
            results["confusion_matrix"],
            output_path=self.REPORT_DIR / "confusion_matrix.png",
        )

        # ------------------------------------------------------
        # SHAP
        # ------------------------------------------------------

        try:

            explainer = SHAPExplainer(model)

            explainer.summary_plot(
                X_test,
                save_path=self.REPORT_DIR / "shap_summary.png",
            )

            logger.info("SHAP summary generated.")

        except Exception as e:

            logger.warning(
                "SHAP explanation skipped: %s",
                e,
            )

        logger.info("=" * 60)
        logger.info("Training Pipeline Completed Successfully")
        logger.info("=" * 60)

        return results


if __name__ == "__main__":

    TrainingPipeline().run()
