from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.auth.service import create_api_key
from app.core.config import Settings
from app.core.model_config import ModelConfig
from app.db.base import Base
from app.db.models import Tenant
from app.main import create_app
from app.providers.base import Provider
from app.providers.fake import FakeProvider
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry

TEST_ALIAS = "test-model"
UPSTREAM_MODEL = "upstream-model"
TEST_DATABASE_URL = "sqlite+aiosqlite://"  # in memory, per engine


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


@pytest.fixture(autouse=True)
def in_memory_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every app built in a test gets its own throwaway database, never ./llmrouter.db."""
    monkeypatch.setenv("LLMROUTER_DATABASE_URL", TEST_DATABASE_URL)


async def seed_credentials(
    app: FastAPI, tenant_name: str = "test-tenant", *, active: bool = True
) -> str:
    """Create the tables, a tenant and an API key in the app's database; return the secret."""
    async with app.state.db_engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with app.state.db_session_factory() as session:
        tenant = Tenant(name=tenant_name, is_active=active)
        session.add(tenant)
        await session.commit()
        _, secret = await create_api_key(session, tenant.id, "test key")
    return secret


class AuthedClient(TestClient):
    """A TestClient that, once started, owns a tenant and sends its API key with every request."""

    secret: str = ""

    def __enter__(self) -> "AuthedClient":
        super().__enter__()
        assert self.portal is not None
        self.secret = self.portal.call(seed_credentials, self.app)
        self.headers["Authorization"] = f"Bearer {self.secret}"
        return self


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def app_(provider: FakeProvider) -> FastAPI:
    def registry_factory(_: httpx.AsyncClient) -> ModelRegistry:
        return registry_with(provider)

    return create_app(
        Settings(_env_file=None, retry_base_delay=0.0),
        registry_factory,
    )


@pytest.fixture
def client(app_: FastAPI) -> Iterator[AuthedClient]:
    with AuthedClient(app_) as test_client:
        yield test_client
