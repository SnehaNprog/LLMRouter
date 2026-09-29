import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import Field, model_validator
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
    redis_key_prefix: str = "llmrouter:"
    # Per-minute allowances for keys that do not set their own; None means unlimited.
    # When Redis (which holds every quota) is unreachable: let requests through unenforced
    # ("open") or refuse requests that have limits ("closed").
    quota_failure_mode: Literal["open", "closed"] = "open"
    quota_breaker_threshold: int = 3  # consecutive failures before Redis is left alone ...
    quota_breaker_cooldown: float = 5.0  # ... for this many seconds
    # Explain each smart-routed answer in x-llmrouter-* response headers. The log always has it.
    routing_headers: bool = True
    default_rpm_limit: int | None = None
    default_tpm_limit: int | None = None
    # Completion tokens assumed when a request sets no max_tokens; settled against real usage.
    default_completion_reserve: int = 512
    http_connect_timeout: float = 5.0
    http_read_timeout: float = 60.0
    http_write_timeout: float = 10.0
    http_pool_timeout: float = 5.0
    retry_max_attempts: int = 3
    retry_base_delay: float = 0.5
    retry_max_delay: float = 8.0
    latency_ewma_alpha: float = 0.2
    # Circuit breakers on every deployment and provider endpoint: open after this many failures
    # in a row (0 turns that rule off) or when this share of at least breaker_min_requests recent
    # calls failed (0 turns that rule off); try again after the cooldown.
    breaker_failure_threshold: int = Field(default=5, ge=0)
    breaker_failure_rate: float = Field(default=0.5, ge=0, le=1)
    breaker_min_requests: int = Field(default=20, ge=1)
    breaker_window_seconds: float = Field(default=60.0, gt=0)
    breaker_cooldown: float = Field(default=30.0, ge=0)
    # After the cooldown, up to breaker_half_open_calls trial calls may be in flight at once and
    # breaker_half_open_successes of them must succeed to close it again. A trial never reported
    # on frees its slot after breaker_probe_timeout seconds.
    breaker_half_open_calls: int = Field(default=1, ge=1)
    breaker_half_open_successes: int = Field(default=1, ge=1)
    breaker_probe_timeout: float = Field(default=30.0, gt=0)
    # Tell the other gateway replicas, through Redis, which breakers are open, and check what they
    # have told us at most every breaker_sync_interval seconds. Redis being down only turns this
    # off: each replica's own breakers keep working.
    breaker_shared: bool = True
    breaker_sync_interval: float = Field(default=1.0, ge=0)
    http_max_connections: int = 100
    http_max_keepalive_connections: int = 20

    @model_validator(mode="after")
    def _a_breaker_rule_is_needed(self) -> "Settings":
        if self.breaker_failure_threshold == 0 and self.breaker_failure_rate == 0:
            raise ValueError(
                "set breaker_failure_threshold or breaker_failure_rate; "
                "with both at 0 no circuit breaker could ever open"
            )
        return self


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
