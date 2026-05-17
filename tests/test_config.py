import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings


def test_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_name == "LLMRouter"
    assert settings.environment == "dev"
    assert settings.log_level == "INFO"


def test_reads_prefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMROUTER_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LLMROUTER_ENVIRONMENT", "prod")

    settings = Settings(_env_file=None)

    assert settings.log_level == "DEBUG"
    assert settings.environment == "prod"


def test_invalid_value_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMROUTER_LOG_LEVEL", "LOUD")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
