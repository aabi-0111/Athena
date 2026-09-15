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

try:
    from app.config import get_config
except ImportError:
    get_config = None  # config module unavailable (e.g. standalone use);
    # DataLoader falls back to DEFAULT_DATASET below rather than failing
    # to import entirely.


logger = get_logger(__name__)


class DataLoader:
    """
    Production-grade dataset loader.

    Path resolution order (first match wins):
    1. `dataset_path` passed explicitly to the constructor.
    2. The `DATA_PATH` value from AppConfig (set via a `DATA_PATH=...`
       line in `.env`, or the `DATA_PATH` environment variable) -- this
       is the recommended way to point at a dataset that lives outside
       the project folder (e.g. on a separate drive), since it means a
       path change is a one-line `.env` edit, never a code edit.
    3. `DEFAULT_DATASET`, a project-relative fallback, used only if
       neither of the above is available (e.g. config couldn't be
       imported at all).
    """

    DEFAULT_DATASET = (
        Path("dataset")
        / "raw"
        / "PS_20174392719_1491204439457_log.csv"
    )

    def __init__(self, dataset_path: Optional[str] = None):

        if dataset_path:
            self.dataset_path = Path(dataset_path)
        else:
            configured_path = self._configured_path()
            self.dataset_path = (
                Path(configured_path) if configured_path else self.DEFAULT_DATASET
            )

    @staticmethod
    def _configured_path() -> Optional[str]:
        """
        Read DATA_PATH from AppConfig, if config is importable.

        Isolated into its own method (rather than inlined in __init__)
        so this is the single place that swallows a config failure --
        a broken/missing .env should degrade to DEFAULT_DATASET, not
        crash DataLoader construction.
        """
        if get_config is None:
            return None
        try:
            return get_config().raw_data_path
        except Exception as exc:
            logger.warning(
                "Could not read DATA_PATH from config (%s); "
                "falling back to DEFAULT_DATASET.", exc,
            )
            return None

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
        