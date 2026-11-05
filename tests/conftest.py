import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import asyncpg
import fakeredis
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncEngine

from app.auth.service import create_api_key
from app.core.config import Settings
from app.core.model_config import ModelConfig
from app.core.redis import build_redis_client
from app.db.base import Base
from app.db.models import Tenant
from app.db.session import create_engine
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


async def _postgres_admin(url: URL, statement: str) -> None:
    connection = await asyncpg.connect(
        user=url.username, password=url.password, host=url.host, port=url.port, database="postgres"
    )
    try:
        await connection.execute(statement)
    finally:
        await connection.close()


def _in_a_thread(coroutine: Awaitable[None]) -> None:
    """Run a coroutine to completion from code that may already be inside an event loop."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(asyncio.run, coroutine).result()  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def throwaway_database(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every app built in a test gets its own throwaway database, never ./llmrouter.db.

    An in-memory SQLite one by default. With LLMROUTER_TEST_POSTGRES_URL set (`make test-postgres`)
    each app instead gets a database created on that server when it starts and dropped when the
    test ends, so the whole suite can also run against Postgres. Tests that name a SQLite file
    themselves keep it.
    """
    server = os.environ.get("LLMROUTER_TEST_POSTGRES_URL")
    if not server:
        monkeypatch.setenv("LLMROUTER_DATABASE_URL", TEST_DATABASE_URL)
        yield
        return
    base = make_url(server).set(drivername="postgresql+asyncpg")
    monkeypatch.setenv("LLMROUTER_DATABASE_URL", base.render_as_string(hide_password=False))
    created: list[str] = []

    def engine_with_own_database(url: str) -> AsyncEngine:
        if (
            make_url(url) != base
        ):  # a URL the test chose itself (a SQLite file, a migrated database)
            return create_engine(url)
        name = f"llmrouter_test_{uuid.uuid4().hex[:12]}"
        _in_a_thread(_postgres_admin(base, f'CREATE DATABASE "{name}"'))
        created.append(name)
        own = make_url(url).set(database=name)
        return create_engine(own.render_as_string(hide_password=False))

    monkeypatch.setattr("app.lifespan.create_engine", engine_with_own_database)
    try:
        yield
    finally:
        for name in created:
            _in_a_thread(_postgres_admin(base, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


SLOW_MS = float(os.environ.get("LLMROUTER_TEST_SLOW_MS", "0") or 0)


@pytest.fixture(autouse=True)
def slow_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the machine is slow: add a delay to every HTTP request and Redis command.

    Off by default. With LLMROUTER_TEST_SLOW_MS=40 (`make test-slow`) each request takes 40 ms
    longer and each Redis command 8 ms, so a test that quietly assumes things happen quickly (a
    token bucket that "refills only a little", a window that "has not passed yet") fails here, on
    purpose and every time, instead of once in a while on a busy CI runner. A test must hold on any
    machine speed to pass in both modes.
    """
    if not SLOW_MS:
        return
    import time

    from redis.asyncio.client import Redis as RedisClient
    from starlette.testclient import _TestClientTransport

    http_delay, redis_delay = SLOW_MS / 1000, SLOW_MS / 5000
    sync_request: Any = _TestClientTransport.handle_request
    async_request: Any = httpx.ASGITransport.handle_async_request
    command: Any = RedisClient.execute_command

    def slow_sync(self: Any, request: httpx.Request) -> httpx.Response:
        time.sleep(http_delay)
        response: httpx.Response = sync_request(self, request)
        return response

    async def slow_async(self: Any, request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(http_delay)
        response: httpx.Response = await async_request(self, request)
        return response

    async def slow_command(self: Any, *args: Any, **options: Any) -> Any:
        await asyncio.sleep(redis_delay)
        return await command(self, *args, **options)

    monkeypatch.setattr(_TestClientTransport, "handle_request", slow_sync)
    monkeypatch.setattr(httpx.ASGITransport, "handle_async_request", slow_async)
    monkeypatch.setattr(RedisClient, "execute_command", slow_command)


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
