"""
Athena v1.0
--------------------
Application Configuration

Responsibilities
----------------
1. Load environment variables (.env) once, at process start.
2. Expose a single validated, immutable config object for the whole app.
3. Let deployment-specific values (ports, thresholds, secrets) be overridden
   without touching code — everything structural/schema-related stays in
   core/constants.py; everything environment-specific lives here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

from dotenv import load_dotenv

from core.constants import (
    PROJECT_ROOT,
    DEFAULT_THRESHOLD,
    RANDOM_STATE,
    ATHENA_LOG_PATH,
    BEST_MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH,
)

# Load .env once at import time. `override=False` means real environment
# variables (e.g. set by the deployment platform) always win over the
# .env file — .env is a local-dev convenience, not a source of truth in
# production.
load_dotenv(PROJECT_ROOT / ".env", override=False)


class Environment(str, Enum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


def _get_str(key: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.getenv(key, default)
    if required and not value:
        raise ConfigError(f"Missing required environment variable: '{key}'")
    return value


def _get_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable '{key}' must be an int, got '{raw}'") from exc


def _get_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable '{key}' must be a float, got '{raw}'") from exc


def _get_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    # --- Environment ---
    environment: Environment
    debug: bool

    # --- API server ---
    api_host: str
    api_port: int

    # --- Logging ---
    log_level: str
    log_file_path: str

    # --- ML ---
    random_state: int
    model_threshold: float
    model_path: str
    scaler_path: str
    encoder_path: str

    # --- Security / compliance ---
    # Salt for the SHA-256 pseudonymization applied to nameOrig/nameDest.
    # Never defaulted to a fixed string — a hardcoded salt defeats the
    # point of salting. Must come from the environment/secrets manager.
    pseudonymization_salt: str = field(repr=False)

    def __post_init__(self) -> None:
        if self.environment == Environment.PROD and self.debug:
            raise ConfigError("debug=True is not allowed when environment=prod.")
        if not (0.0 < self.model_threshold < 1.0):
            raise ConfigError(
                f"model_threshold must be in (0, 1), got {self.model_threshold}."
            )


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
    Build (once) and return the process-wide config.

    Cached with `lru_cache` rather than a bare module-level global so
    that (a) construction — including validation in `__post_init__` — is
    lazy, not paid at import time, and (b) tests can call
    `get_config.cache_clear()` to rebuild against a patched environment
    instead of reloading the module.
    """
    environment = Environment(_get_str("ATHENA_ENV", default="dev").lower())

    return AppConfig(
        environment=environment,
        debug=_get_bool("ATHENA_DEBUG", default=environment != Environment.PROD),
        api_host=_get_str("API_HOST", default="0.0.0.0"),
        api_port=_get_int("API_PORT", default=8000),
        log_level=_get_str("LOG_LEVEL", default="INFO").upper(),
        log_file_path=_get_str("LOG_FILE_PATH", default=str(ATHENA_LOG_PATH)),
        random_state=_get_int("RANDOM_STATE", default=RANDOM_STATE),
        model_threshold=_get_float("MODEL_THRESHOLD", default=DEFAULT_THRESHOLD),
        model_path=_get_str("MODEL_PATH", default=str(BEST_MODEL_PATH)),
        scaler_path=_get_str("SCALER_PATH", default=str(SCALER_PATH)),
        encoder_path=_get_str("ENCODER_PATH", default=str(ENCODER_PATH)),
        pseudonymization_salt=_get_str(
            "PSEUDONYMIZATION_SALT",
            default="dev-only-insecure-salt" if environment == Environment.DEV else None,
            required=environment != Environment.DEV,
        ),
    )
