from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine as create_sync_engine
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.db.base import Base
from app.db.models import ApiKey, Tenant
from app.db.session import create_engine, create_session_factory, database_ready

ROOT = Path(__file__).parent.parent


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    engine = create_engine("sqlite+aiosqlite://")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    async with create_session_factory(engine)() as session:
        yield session


def make_key(tenant_id: str, key_hash: str = "h1") -> ApiKey:
    return ApiKey(tenant_id=tenant_id, name="ci", key_prefix="llmr_ab", key_hash=key_hash)


# ---------- models ----------


async def test_tenant_gets_defaults(session: AsyncSession) -> None:
    tenant = Tenant(name="acme")
    session.add(tenant)
    await session.commit()

    assert len(tenant.id) == 32
    assert tenant.is_active is True
    assert tenant.created_at.tzinfo is not None
    assert abs(datetime.now(UTC) - tenant.created_at) < timedelta(seconds=5)


async def test_key_belongs_to_a_tenant(session: AsyncSession) -> None:
    tenant = Tenant(name="acme")
    tenant.api_keys.append(make_key("ignored"))
    session.add(tenant)
    await session.commit()

    stored = (await session.execute(select(ApiKey))).scalar_one()
    assert stored.tenant_id == tenant.id
    assert stored.last_used_at is None
    assert stored.created_at.tzinfo is not None


async def test_tenant_names_are_unique(session: AsyncSession) -> None:
    session.add_all([Tenant(name="acme"), Tenant(name="acme")])

    with pytest.raises(IntegrityError):
        await session.commit()


async def test_key_hashes_are_unique(session: AsyncSession) -> None:
    tenant = Tenant(name="acme")
    session.add(tenant)
    await session.commit()
    session.add_all([make_key(tenant.id, "same"), make_key(tenant.id, "same")])

    with pytest.raises(IntegrityError):
        await session.commit()


async def test_key_for_a_missing_tenant_is_rejected(session: AsyncSession) -> None:
    session.add(make_key("no-such-tenant"))

    with pytest.raises(IntegrityError):
        await session.commit()


async def test_deleting_a_tenant_deletes_its_keys(session: AsyncSession) -> None:
    tenant = Tenant(name="acme")
    other = Tenant(name="globex")
    session.add_all([tenant, other])
    await session.commit()
    session.add_all([make_key(tenant.id, "a"), make_key(other.id, "b")])
    await session.commit()

    await session.delete(tenant)
    await session.commit()

    remaining = (await session.execute(select(ApiKey))).scalars().all()
    assert [k.key_hash for k in remaining] == ["b"]


# ---------- UTC datetimes ----------


async def test_datetimes_are_stored_as_utc(session: AsyncSession) -> None:
    plus_five = timezone(timedelta(hours=5))
    moment = datetime(2026, 1, 1, 12, 0, tzinfo=plus_five)
    tenant = Tenant(name="acme", created_at=moment)
    session.add(tenant)
    await session.commit()
    session.expire_all()

    stored = (await session.execute(select(Tenant))).scalar_one()

    assert stored.created_at == moment
    assert stored.created_at.utcoffset() == timedelta(0)
    assert stored.created_at.hour == 7


async def test_naive_datetimes_are_rejected(session: AsyncSession) -> None:
    session.add(Tenant(name="acme", created_at=datetime(2026, 1, 1)))  # noqa: DTZ001

    with pytest.raises(Exception, match="naive datetimes"):
        await session.commit()


# ---------- connection helpers ----------


async def test_database_ready_when_reachable(engine: AsyncEngine) -> None:
    assert await database_ready(engine) is True


async def test_database_not_ready_when_unreachable(tmp_path: Path) -> None:
    broken = create_engine(f"sqlite+aiosqlite:///{tmp_path}/missing-dir/db.sqlite")

    try:
        assert await database_ready(broken) is False
    finally:
        await broken.dispose()


async def test_foreign_keys_are_switched_on(engine: AsyncEngine) -> None:
    async with engine.connect() as connection:
        result = await connection.exec_driver_sql("PRAGMA foreign_keys")

        assert result.scalar_one() == 1


# ---------- migrations ----------


def alembic_config(database: Path) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{database}")
    return config


def table_names(database: Path) -> set[str]:
    engine = create_sync_engine(f"sqlite:///{database}")
    try:
        return set(inspect(engine).get_table_names()) - {"alembic_version"}
    finally:
        engine.dispose()


def test_migrations_create_the_tables(tmp_path: Path) -> None:
    database = tmp_path / "migrated.db"

    command.upgrade(alembic_config(database), "head")

    assert table_names(database) == {"tenants", "api_keys", "request_logs"}


def test_migrations_match_the_models(tmp_path: Path) -> None:
    """If this fails, a model changed without a migration (run `make migration m=...`)."""
    database = tmp_path / "migrated.db"
    command.upgrade(alembic_config(database), "head")

    engine = create_sync_engine(f"sqlite:///{database}")
    try:
        with engine.connect() as connection:
            diff = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    finally:
        engine.dispose()

    assert diff == []


def test_migrations_can_be_reversed(tmp_path: Path) -> None:
    database = tmp_path / "migrated.db"
    config = alembic_config(database)

    command.upgrade(config, "head")
    command.downgrade(config, "base")

    assert table_names(database) == set()


def test_upgrading_twice_is_harmless(tmp_path: Path) -> None:
    database = tmp_path / "migrated.db"
    config = alembic_config(database)

    command.upgrade(config, "head")
    command.upgrade(config, "head")

    assert table_names(database) == {"tenants", "api_keys", "request_logs"}


async def test_session_factory_commits_across_sessions(engine: AsyncEngine) -> None:
    factory = create_session_factory(engine)
    async with factory() as first:
        first.add(Tenant(name="acme"))
        await first.commit()

    async with factory() as second:
        names = (await second.execute(select(Tenant.name))).scalars().all()

    assert names == ["acme"]
