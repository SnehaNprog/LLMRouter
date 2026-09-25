import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

import fakeredis
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from redis.asyncio import Redis
from sqlalchemy import select

from app.auth.service import create_api_key
from app.core.config import Settings
from app.core.model_config import ModelConfig
from app.core.redis import build_redis_client
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


@pytest.fixture(autouse=True)
def fake_redis_for_apps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apps built in tests get a private in-memory Redis, never a real server."""
    monkeypatch.setattr(
        "app.lifespan.build_redis_client",
        lambda settings: fakeredis.FakeAsyncRedis(decode_responses=True),
    )


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

    def issue_key(
        self,
        scopes: list[str] | None = None,
        allowed_models: list[str] | None = None,
        rpm_limit: int | None = None,
        tpm_limit: int | None = None,
        monthly_budget_usd: float | None = None,
    ) -> str:
        """Create another key for the same tenant; return its secret."""

        async def create() -> str:
            async with self.app.state.db_session_factory() as session:  # type: ignore[attr-defined]
                tenant = (await session.execute(select(Tenant))).scalars().first()
                assert tenant is not None
                _, secret = await create_api_key(
                    session,
                    tenant.id,
                    "extra key",
                    scopes=scopes,
                    allowed_models=allowed_models,
                    rpm_limit=rpm_limit,
                    tpm_limit=tpm_limit,
                    monthly_budget_usd=monthly_budget_usd,
                )
                return str(secret)

        assert self.portal is not None
        return str(self.portal.call(create))

    def as_key(self, secret: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {secret}"}


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


@pytest.fixture(params=["fake", "real"])
async def redis_client(request: pytest.FixtureRequest) -> AsyncIterator[Redis]:
    """The same Redis tests, run against an in-memory fake and (opt-in) a real server.

    The real server is used only when LLMROUTER_TEST_REDIS_URL is set, because the tests flush
    the database it points at: use a database you do not care about (`make test-redis`).
    """
    client: Redis
    if request.param == "fake":
        client = fakeredis.FakeAsyncRedis(decode_responses=True)
    else:
        url = os.environ.get("LLMROUTER_TEST_REDIS_URL")
        if not url:
            pytest.skip("set LLMROUTER_TEST_REDIS_URL to also test against a real Redis")
        client = build_redis_client(Settings(_env_file=None, redis_url=url))  # as in production
        await client.ping()  # opted in, so an unreachable server is a failure, not a skip
        await client.flushdb()
    yield client
    if request.param == "real":
        await client.flushdb()
    await client.aclose()
