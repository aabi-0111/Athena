"""
Data loading module for Athena.

Responsible for:

- Loading PaySim dataset
- Dataset validation
- Memory optimization
- Basic metadata logging
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from app.core.logger import get_logger


logger = get_logger(__name__)


class DataLoader:
    """
    Production-grade dataset loader.
    """

    DEFAULT_DATASET = (
        Path("dataset")
        / "raw"
        / "PS_20174392719_1491204439457_log.csv"
    )

    def __init__(self, dataset_path: Optional[str] = None):

        self.dataset_path = (
            Path(dataset_path)
            if dataset_path
            else self.DEFAULT_DATASET
        )

    def load(self) -> pd.DataFrame:
        """
        Load dataset.

        Returns
        -------
        pd.DataFrame
        """

        logger.info("Loading dataset...")

        self._validate_path()

        df = pd.read_csv(
            self.dataset_path,
            low_memory=False,
        )

        logger.info(
            "Dataset loaded successfully | Rows=%d Columns=%d",
            df.shape[0],
            df.shape[1],
        )

        return df

    def _validate_path(self) -> None:

        if not self.dataset_path.exists():
            logger.error("Dataset not found.")

            raise FileNotFoundError(
                f"Dataset does not exist:\n{self.dataset_path}"
            )

        logger.info("Dataset located.")

    @staticmethod
    def dataset_info(df: pd.DataFrame) -> None:

        logger.info("=" * 60)
        logger.info("Dataset Information")
        logger.info("=" * 60)

        logger.info("Rows      : %d", df.shape[0])
        logger.info("Columns   : %d", df.shape[1])
        logger.info("Memory(MB): %.2f",
                    df.memory_usage(deep=True).sum() / 1024**2)

        logger.info("Missing Values:\n%s",
                    df.isnull().sum())

        logger.info("=" * 60)
        