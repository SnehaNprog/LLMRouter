from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLMROUTER_", env_file=".env", extra="ignore")

    app_name: str = "LLMRouter"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    http_connect_timeout: float = 5.0
    http_read_timeout: float = 60.0
    http_write_timeout: float = 10.0
    http_pool_timeout: float = 5.0
    http_max_connections: int = 100
    http_max_keepalive_connections: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
