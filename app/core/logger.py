"""
Centralized logging configuration for Athena.
"""

from pathlib import Path
from logging.handlers import RotatingFileHandler
import logging

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "athena.log"

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | "
    "%(name)s | %(filename)s:%(lineno)d | %(message)s"
)

_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
