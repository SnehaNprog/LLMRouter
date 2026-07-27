import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, provider_environment
from app.core.model_config import ConfigError
from app.main import create_app

GOOD_YAML = """\
models:
  - name: fast
    provider: openai_compatible
    model: llama-3.1-8b-instant
    base_url: https://api.groq.com/openai/v1
    api_key: os.environ/STARTUP_TEST_KEY
  - name: fast
    provider: openai_compatible
    model: llama-3.1-70b
    base_url: https://api.groq.com/openai/v1
    api_key: os.environ/STARTUP_TEST_KEY
  - name: local
    provider: openai_compatible
    model: llama3.2
    base_url: http://localhost:11434/v1
"""


def app_for(path: Path):  # type: ignore[no-untyped-def]
    return create_app(Settings(_env_file=None, config_path=path))


def test_valid_config_loads_and_serves_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(GOOD_YAML)
    monkeypatch.setenv("STARTUP_TEST_KEY", "secret")

    with TestClient(app_for(config)) as client:
        ids = [m["id"] for m in client.get("/v1/models").json()["data"]]

    assert ids == ["fast", "local"]


def test_missing_config_file_prevents_startup(tmp_path: Path) -> None:
    app = app_for(tmp_path / "nope.yaml")

    with pytest.raises(ConfigError, match="cannot read model config"), TestClient(app):
        pass


def test_missing_secret_prevents_startup_and_names_the_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(GOOD_YAML)
    monkeypatch.delenv("STARTUP_TEST_KEY", raising=False)

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(ConfigError, match="STARTUP_TEST_KEY is not set"),
        TestClient(app_for(config)),
    ):
        pass

    assert any("startup aborted" in record.getMessage() for record in caplog.records)


def test_invalid_config_prevents_startup(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("models:\n  - name: x\n    provider: gemini\n    model: m\n")

    with pytest.raises(ConfigError, match=r"models\[0\]\.provider"), TestClient(app_for(config)):
        pass


def test_http_client_is_closed_even_when_startup_fails(tmp_path: Path) -> None:
    app = app_for(tmp_path / "nope.yaml")

    with pytest.raises(ConfigError), TestClient(app):
        pass

    assert app.state.http_client.is_closed


# ---------- .env handling for provider keys ----------


def test_provider_environment_reads_dotenv_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ONLY_IN_DOTENV=from-file\nSHARED=from-file\n")
    monkeypatch.delenv("ONLY_IN_DOTENV", raising=False)
    monkeypatch.setenv("SHARED", "from-environment")

    values = provider_environment(env_file)

    assert values["ONLY_IN_DOTENV"] == "from-file"
    assert values["SHARED"] == "from-environment"


def test_provider_environment_without_dotenv_file(tmp_path: Path) -> None:
    values = provider_environment(tmp_path / "missing.env")

    assert "PATH" in values
