import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLMROUTER_", env_file=".env", extra="ignore")

    app_name: str = "LLMRouter"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    config_path: Path = Path("config.yaml")
    database_url: str = "sqlite+aiosqlite:///./llmrouter.db"
    redis_url: str = "redis://localhost:6379/0"
    redis_timeout: float = (
        1.0  # seconds: connecting, each command, and waiting for a free connection
    )
    redis_max_connections: int = 50
    http_connect_timeout: float = 5.0
    http_read_timeout: float = 60.0
    http_write_timeout: float = 10.0
    http_pool_timeout: float = 5.0
    retry_max_attempts: int = 3
    retry_base_delay: float = 0.5
    retry_max_delay: float = 8.0
    latency_ewma_alpha: float = 0.2
    http_max_connections: int = 100
    http_max_keepalive_connections: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()


def provider_environment(env_file: Path = Path(".env")) -> dict[str, str]:
    """Variables that ``os.environ/NAME`` references in the model config may resolve against.

    pydantic-settings reads .env without exporting it, so provider keys kept there would
    otherwise be invisible. Real environment variables win over .env entries.
    """
    values: dict[str, str] = {}
    if env_file.is_file():
        values.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
    values.update(os.environ)
    return values
