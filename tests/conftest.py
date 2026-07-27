from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import Settings
from app.core.model_config import ModelConfig
from app.main import create_app
from app.providers.base import Provider
from app.providers.fake import FakeProvider
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry

TEST_ALIAS = "test-model"
UPSTREAM_MODEL = "upstream-model"


def model_config(
    name: str = TEST_ALIAS, model: str = UPSTREAM_MODEL, **overrides: object
) -> ModelConfig:
    fields: dict[str, Any] = {
        "name": name,
        "provider": "openai_compatible",
        "model": model,
        "base_url": "https://api.example.com/v1",
        "api_key": SecretStr("test-key"),
        "weight": 1,
        "rpm": None,
        "tpm": None,
        "api_key_env": "TEST_KEY",
    }
    fields.update(overrides)
    return ModelConfig(**fields)


def registry_with(provider: Provider, **config_overrides: Any) -> ModelRegistry:
    return ModelRegistry([Deployment.create(model_config(**config_overrides), provider)])


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def app_(provider: FakeProvider) -> FastAPI:
    def registry_factory(_: httpx.AsyncClient) -> ModelRegistry:
        return registry_with(provider)

    return create_app(Settings(_env_file=None), registry_factory)


@pytest.fixture
def client(app_: FastAPI) -> Iterator[TestClient]:
    with TestClient(app_) as test_client:
        yield test_client
