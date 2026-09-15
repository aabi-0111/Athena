"""Dataset loading utilities for Athena."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from app.core.constants import RAW_DATA_PATH
from app.core.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Load and validate the configured raw dataset."""

    DEFAULT_DATASET = RAW_DATA_PATH

    def __init__(self, dataset_path: Optional[str | Path] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else self.DEFAULT_DATASET

    def load(self) -> pd.DataFrame:
        logger.info("Loading dataset...")
        self._validate_path()
        df = pd.read_csv(self.dataset_path, low_memory=False)
        logger.info("Dataset loaded successfully | Rows=%d Columns=%d", df.shape[0], df.shape[1])
        return df

    def _validate_path(self) -> None:
        if not self.dataset_path.exists():
            logger.error("Dataset not found: %s", self.dataset_path)
            raise FileNotFoundError(f"Dataset does not exist:\n{self.dataset_path}")
        logger.info("Dataset located: %s", self.dataset_path)

    @staticmethod
    def dataset_info(df: pd.DataFrame) -> None:
        logger.info("=" * 60)
        logger.info("Dataset Information")
        logger.info("=" * 60)
        logger.info("Rows      : %d", df.shape[0])
        logger.info("Columns   : %d", df.shape[1])
        logger.info("Memory(MB): %.2f", df.memory_usage(deep=True).sum() / 1024**2)
        logger.info("Missing Values:\n%s", df.isnull().sum())
        logger.info("=" * 60)
