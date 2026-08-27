import json
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic import command
from fastapi import Depends, FastAPI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.auth.dependencies import AuthContext, authenticate
from app.auth.scopes import (
    ADMIN_KEYS,
    ADMIN_TENANTS,
    CHAT_WRITE,
    DEFAULT_SCOPES,
    KNOWN_SCOPES,
    MODELS_READ,
    normalize_scopes,
    scope_allows,
)
from app.auth.service import create_api_key, find_key_by_secret, rotate_api_key
from app.db.base import Base
from app.db.models import ApiKey, Tenant
from app.db.session import create_engine, create_session_factory
from tests.conftest import AuthedClient
from tests.test_db import alembic_config

# ---------- vocabulary ----------


def test_known_scopes() -> None:
    assert KNOWN_SCOPES == {"chat:write", "models:read", "admin:tenants", "admin:keys"}


def test_defaults_are_least_privilege() -> None:
    assert set(DEFAULT_SCOPES) == {CHAT_WRITE, MODELS_READ}
    assert not any(scope.startswith("admin:") for scope in DEFAULT_SCOPES)


def test_known_scopes_are_accepted() -> None:
    assert normalize_scopes([CHAT_WRITE, MODELS_READ]) == ["chat:write", "models:read"]


@pytest.mark.parametrize("wildcard", ["chat:*", "models:*", "admin:*"])
def test_resource_wildcards_are_accepted(wildcard: str) -> None:
    assert normalize_scopes([wildcard]) == [wildcard]


def test_scopes_are_deduplicated_and_sorted() -> None:
    assert normalize_scopes(["models:read", "chat:write", "models:read"]) == [
        "chat:write",
        "models:read",
    ]


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "*",
        "chat",
        "chat:read",
        "CHAT:WRITE",
        "chat:write:extra",
        ":write",
        "foo:*",
        "admin:root",
    ],
)
def test_invalid_scopes_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="unknown scope"):
        normalize_scopes([CHAT_WRITE, bad])


def test_the_error_lists_every_bad_scope_and_the_valid_ones() -> None:
    with pytest.raises(ValueError) as excinfo:
        normalize_scopes(["nope", "chat:write", "also:nope"])

    message = str(excinfo.value)
    assert "also:nope" in message
    assert "nope" in message
    assert "admin:keys" in message


def test_a_key_needs_at_least_one_scope() -> None:
    with pytest.raises(ValueError, match="at least one"):
        normalize_scopes([])


# ---------- matching ----------


def test_exact_scope_matches() -> None:
    assert scope_allows([CHAT_WRITE], CHAT_WRITE)


def test_missing_scope_does_not_match() -> None:
    assert not scope_allows([MODELS_READ], CHAT_WRITE)
    assert not scope_allows([], CHAT_WRITE)


def test_wildcard_grants_every_action_on_its_resource() -> None:
    assert scope_allows(["admin:*"], ADMIN_TENANTS)
    assert scope_allows(["admin:*"], ADMIN_KEYS)


def test_wildcard_does_not_cross_resources() -> None:
    assert not scope_allows(["admin:*"], CHAT_WRITE)
    assert not scope_allows(["chat:*"], ADMIN_KEYS)


def test_admin_keys_do_not_get_chat_by_accident() -> None:
    assert not scope_allows([ADMIN_TENANTS, ADMIN_KEYS], CHAT_WRITE)


def test_asking_for_an_unknown_scope_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="not a known scope"):
        scope_allows(["chat:write"], "chat:writ")


# ---------- stored on keys ----------


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


@pytest.fixture
async def tenant(session: AsyncSession) -> Tenant:
    tenant = Tenant(name="acme")
    session.add(tenant)
    await session.commit()
    return tenant


async def test_new_keys_get_the_default_scopes(session: AsyncSession, tenant: Tenant) -> None:
    key, _ = await create_api_key(session, tenant.id, "ci")

    assert key.scopes == ["chat:write", "models:read"]


async def test_custom_scopes_are_stored_normalized(session: AsyncSession, tenant: Tenant) -> None:
    key, _ = await create_api_key(
        session, tenant.id, "ops", scopes=["admin:keys", "admin:tenants", "admin:keys"]
    )

    assert key.scopes == ["admin:keys", "admin:tenants"]


async def test_scopes_survive_a_round_trip(session: AsyncSession, tenant: Tenant) -> None:
    _, secret = await create_api_key(session, tenant.id, "ops", scopes=["models:read"])
    session.expire_all()

    found = await find_key_by_secret(session, secret)

    assert found is not None
    assert found.scopes == ["models:read"]


async def test_invalid_scopes_create_nothing(session: AsyncSession, tenant: Tenant) -> None:
    with pytest.raises(ValueError, match="unknown scope"):
        await create_api_key(session, tenant.id, "ci", scopes=["root"])
    with pytest.raises(ValueError, match="at least one"):
        await create_api_key(session, tenant.id, "ci", scopes=[])

    assert (await session.execute(select(ApiKey))).first() is None


async def test_rotation_keeps_the_scopes(session: AsyncSession, tenant: Tenant) -> None:
    old, _ = await create_api_key(session, tenant.id, "ops", scopes=["admin:*", "models:read"])

    new, _ = await rotate_api_key(session, tenant.id, old.id)

    assert new.scopes == ["admin:*", "models:read"]


async def test_rotation_can_change_the_scopes(session: AsyncSession, tenant: Tenant) -> None:
    old, _ = await create_api_key(session, tenant.id, "ops", scopes=["admin:*"])

    new, _ = await rotate_api_key(session, tenant.id, old.id, scopes=["models:read"])

    assert new.scopes == ["models:read"]
    assert old.scopes == ["admin:*"]  # the old key is untouched, only retired


# ---------- visible to request handling ----------


def test_authentication_exposes_the_key_scopes(client: AuthedClient, app_: FastAPI) -> None:
    @app_.get("/whoami")
    def whoami(auth: AuthContext = Depends(authenticate)) -> dict[str, object]:  # noqa: B008
        return {"tenant": auth.tenant_name, "scopes": sorted(auth.scopes)}

    body = client.get("/whoami").json()

    assert body == {"tenant": "test-tenant", "scopes": ["chat:write", "models:read"]}


# ---------- migration ----------


def test_migration_gives_existing_keys_the_default_scopes(tmp_path: Path) -> None:
    database = tmp_path / "scopes.db"
    config = alembic_config(database)
    command.upgrade(config, "0002")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO tenants (id, name, is_active, created_at)"
            " VALUES ('t1', 'acme', 1, '2026-01-01')"
        )
        connection.execute(
            "INSERT INTO api_keys (id, tenant_id, name, key_prefix, key_hash, created_at)"
            " VALUES ('k1', 't1', 'old key', 'llmr_abc', 'hash', '2026-01-01')"
        )

    command.upgrade(config, "head")

    with sqlite3.connect(database) as connection:
        (stored,) = connection.execute("SELECT scopes FROM api_keys WHERE id = 'k1'").fetchone()
    assert json.loads(stored) == ["chat:write", "models:read"]


def test_scope_migration_can_be_reversed(tmp_path: Path) -> None:
    database = tmp_path / "scopes.db"
    config = alembic_config(database)
    command.upgrade(config, "head")

    command.downgrade(config, "0002")

    with sqlite3.connect(database) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(api_keys)")]
    assert "scopes" not in columns
    assert "expires_at" in columns
