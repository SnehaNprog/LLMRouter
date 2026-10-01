"""Checks that only mean something on a real Postgres. Skipped unless LLMROUTER_TEST_POSTGRES_URL
is set (`make test-postgres`); the rest of the suite also runs on it then, one database per app."""

import asyncio
import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import func, inspect, select
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import Settings
from app.db.base import Base
from app.db.models import RequestLog, Tenant, new_id
from app.db.session import create_engine, create_session_factory
from app.main import create_app
from app.providers.fake import FakeProvider
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry
from tests.conftest import TEST_ALIAS, AuthedClient, _postgres_admin, model_config, seed_credentials
from tests.test_db import ROOT

SERVER = os.environ.get("LLMROUTER_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not SERVER, reason="set LLMROUTER_TEST_POSTGRES_URL to also test against Postgres"
)

BODY = {"model": TEST_ALIAS, "messages": [{"role": "user", "content": "hello there"}]}


def a_log(tenant_id: str, created: datetime) -> RequestLog:
    return RequestLog(
        id=new_id(),
        created_at=created,
        day=created.strftime("%Y-%m-%d"),
        tenant_id=tenant_id,
        key_id="k",
        requested_model="m",
        status_code=200,
        stream=False,
        prompt_tokens=1,
        completion_tokens=1,
        cost_nano_usd=1,
        latency_ms=1,
        routed=False,
        escalated=False,
    )


@pytest.fixture
def database() -> Iterator[URL]:
    """A new, empty database on the server, dropped afterwards."""
    assert SERVER is not None
    base = make_url(SERVER).set(drivername="postgresql+asyncpg")
    name = f"llmrouter_test_{uuid.uuid4().hex[:12]}"
    asyncio.run(_postgres_admin(base, f'CREATE DATABASE "{name}"'))
    try:
        yield base.set(database=name)
    finally:
        asyncio.run(_postgres_admin(base, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


def config_for(url: URL) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", url.render_as_string(hide_password=False))
    return config


async def table_names(url: URL) -> set[str]:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            names = await connection.run_sync(lambda c: set(inspect(c).get_table_names()))
        return names - {"alembic_version"}
    finally:
        await engine.dispose()


def tables(url: URL) -> set[str]:
    return asyncio.run(table_names(url))


# ---------- this suite really is running on Postgres ----------


def test_apps_in_this_run_use_postgres() -> None:
    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return ModelRegistry([Deployment.create(model_config(), FakeProvider())])

    with AuthedClient(create_app(Settings(_env_file=None), factory)) as client:
        assert client.app.state.db_engine.dialect.name == "postgresql"  # type: ignore[attr-defined]


# ---------- migrations ----------


def test_migrations_create_the_tables(database: URL) -> None:
    command.upgrade(config_for(database), "head")
    assert tables(database) == {"tenants", "api_keys", "request_logs"}


def test_migrations_match_the_models(database: URL) -> None:
    """If this fails, a model changed without a migration, or one is not portable to Postgres."""
    command.upgrade(config_for(database), "head")

    async def drift() -> list[Any]:
        engine = create_async_engine(database)
        try:
            async with engine.connect() as connection:
                return await connection.run_sync(
                    lambda c: compare_metadata(MigrationContext.configure(c), Base.metadata)
                )
        finally:
            await engine.dispose()

    assert asyncio.run(drift()) == []


def test_every_migration_applies_and_reverses_one_step_at_a_time(database: URL) -> None:
    config = config_for(database)
    revisions = [r.revision for r in ScriptDirectory.from_config(config).walk_revisions()][::-1]
    assert len(revisions) >= 9

    for revision in revisions:
        command.upgrade(config, revision)
    assert tables(database) == {"tenants", "api_keys", "request_logs"}
    for _ in revisions:
        command.downgrade(config, "-1")  # each step down must succeed too

    assert tables(database) == set()


def test_upgrading_twice_is_harmless(database: URL) -> None:
    config = config_for(database)
    command.upgrade(config, "head")
    command.upgrade(config, "head")
    assert tables(database) == {"tenants", "api_keys", "request_logs"}


def test_a_database_made_by_migrations_serves_traffic(database: URL) -> None:
    """The deployed path: tables come from `alembic upgrade head`, never from create_all."""
    command.upgrade(config_for(database), "head")

    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return ModelRegistry([Deployment.create(model_config(), FakeProvider())])

    settings = Settings(_env_file=None, database_url=database.render_as_string(hide_password=False))

    async def run() -> int:
        app = create_app(settings, factory)
        async with app.router.lifespan_context(app):
            secret = await seed_migrated(app)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://gw",
                headers={"Authorization": f"Bearer {secret}"},
            ) as client:
                response = await client.post("/v1/chat/completions", json=BODY)
            async with app.state.db_session_factory() as session:
                logged = (
                    await session.execute(select(func.count()).select_from(RequestLog))
                ).scalar_one()
        assert response.status_code == 200
        return int(logged)

    assert asyncio.run(run()) == 1


async def seed_migrated(app: Any) -> str:
    """Like seed_credentials, but the tables already exist (made by the migrations)."""
    from app.auth.service import create_api_key

    async with app.state.db_session_factory() as session:
        tenant = Tenant(name="deployed")
        session.add(tenant)
        await session.commit()
        _, secret = await create_api_key(session, tenant.id, "key")
    return str(secret)


# ---------- behaviour that differs from SQLite ----------


def test_timestamps_keep_their_zone_and_their_order() -> None:
    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return ModelRegistry([Deployment.create(model_config(), FakeProvider())])

    async def run() -> list[datetime]:
        app = create_app(Settings(_env_file=None), factory)
        async with app.router.lifespan_context(app):
            await seed_credentials(app)
            async with app.state.db_session_factory() as session:
                tenant = (await session.execute(select(Tenant))).scalars().one()
                start = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
                for n in range(3):
                    session.add(a_log(tenant.id, start + timedelta(minutes=n)))
                await session.commit()
                result = await session.execute(
                    select(RequestLog.created_at).order_by(RequestLog.created_at)
                )
                return list(result.scalars())

    stamps = asyncio.run(run())
    assert [s.tzinfo is not None for s in stamps] == [True, True, True]
    assert stamps == sorted(stamps) and stamps[1] - stamps[0] == timedelta(minutes=1)


def test_sums_of_large_amounts_come_back_exact() -> None:
    """Postgres sums BigInteger columns as Decimal and they can pass 2**31; nothing may be lost."""
    from tests.test_usage_api import log_row, my_tenant, new_client, reader, seed

    with new_client() as client:
        tenant = my_tenant(client)
        big = 4_000_000_000_000  # $4000 in nano-dollars: far beyond a 32-bit integer
        seed(
            client,
            *[log_row(tenant, cost_nano_usd=big, prompt_tokens=1_000_000_000) for _ in range(3)],
        )
        response = client.get(
            "/v1/usage", params={"start": "2026-10-01T00:00:00Z"}, headers=reader(client)
        )

    totals = response.json()["totals"]
    assert totals["cost_usd"] == 12_000.0
    assert totals["prompt_tokens"] == 3_000_000_000  # past what a 32-bit integer can hold


async def test_many_slow_requests_at_once_do_not_starve_the_pool(database: URL) -> None:
    """The auth connection must be returned before the model is called, here with a real pool."""
    provider = FakeProvider(latency=0.1)

    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return ModelRegistry([Deployment.create(model_config(), provider)])

    settings = Settings(_env_file=None, database_url=database.render_as_string(hide_password=False))
    app = create_app(settings, factory)
    async with app.router.lifespan_context(app):
        secret = await seed_credentials(app)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gw",
            headers={"Authorization": f"Bearer {secret}"},
        ) as client:
            responses = await asyncio.wait_for(
                asyncio.gather(
                    *[client.post("/v1/chat/completions", json=BODY) for _ in range(60)]
                ),
                timeout=30,
            )
        async with app.state.db_session_factory() as session:
            count = (
                await session.execute(select(func.count()).select_from(RequestLog))
            ).scalar_one()

    assert {r.status_code for r in responses} == {200}
    assert count == 60


async def test_foreign_keys_cascade_when_a_tenant_is_deleted(database: URL) -> None:
    engine = create_engine(database.render_as_string(hide_password=False))
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        factory = create_session_factory(engine)
        async with factory() as session:
            tenant = Tenant(name="doomed")
            session.add(tenant)
            await session.commit()
            session.add(a_log(tenant.id, datetime.now(UTC)))
            await session.commit()
            await session.delete(tenant)
            await session.commit()
            left = (
                await session.execute(select(func.count()).select_from(RequestLog))
            ).scalar_one()
        assert left == 0
    finally:
        await engine.dispose()
