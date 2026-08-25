import sqlite3
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

import pytest
from alembic import command
from fastapi import FastAPI
from openai import AuthenticationError, OpenAI
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.auth.service import (
    KeyInactiveError,
    KeyNotFoundError,
    create_api_key,
    find_key_by_secret,
    revoke_api_key,
    rotate_api_key,
)
from app.db.base import Base
from app.db.models import ApiKey, Tenant
from app.db.session import create_engine, create_session_factory
from tests.conftest import AuthedClient
from tests.test_db import alembic_config

T = TypeVar("T")
HOUR = timedelta(hours=1)


def now() -> datetime:
    return datetime.now(UTC)


# ---------- service level ----------


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


async def other_tenant(session: AsyncSession) -> Tenant:
    other = Tenant(name="globex")
    session.add(other)
    await session.commit()
    return other


async def test_new_keys_are_active_and_never_expire_by_default(
    session: AsyncSession, tenant: Tenant
) -> None:
    key, _ = await create_api_key(session, tenant.id, "ci")

    assert key.status() == "active"
    assert key.expires_at is None
    assert key.revoked_at is None


async def test_expiry_can_be_set_at_creation(session: AsyncSession, tenant: Tenant) -> None:
    deadline = now() + HOUR

    key, _ = await create_api_key(session, tenant.id, "ci", expires_at=deadline)

    assert key.expires_at == deadline
    assert key.status() == "active"


async def test_expiry_must_be_in_the_future_and_timezone_aware(
    session: AsyncSession, tenant: Tenant
) -> None:
    with pytest.raises(ValueError, match="future"):
        await create_api_key(session, tenant.id, "ci", expires_at=now() - HOUR)
    with pytest.raises(ValueError, match="timezone-aware"):
        await create_api_key(session, tenant.id, "ci", expires_at=datetime(2999, 1, 1))  # noqa: DTZ001


def test_status_transitions() -> None:
    moment = datetime(2030, 1, 1, tzinfo=UTC)
    key = ApiKey(expires_at=moment)

    assert key.status(moment - timedelta(seconds=1)) == "active"
    assert key.status(moment) == "expired"  # expired from the very instant
    assert key.status(moment + HOUR) == "expired"


def test_revocation_wins_over_expiry() -> None:
    key = ApiKey(
        expires_at=datetime(2030, 1, 1, tzinfo=UTC), revoked_at=datetime(2029, 1, 1, tzinfo=UTC)
    )

    assert key.status(datetime(2031, 1, 1, tzinfo=UTC)) == "revoked"


async def test_revoke_marks_the_key(session: AsyncSession, tenant: Tenant) -> None:
    key, _ = await create_api_key(session, tenant.id, "ci")

    revoked = await revoke_api_key(session, tenant.id, key.id)

    assert revoked.status() == "revoked"
    assert revoked.revoked_at is not None


async def test_revoking_twice_keeps_the_first_timestamp(
    session: AsyncSession, tenant: Tenant
) -> None:
    key, _ = await create_api_key(session, tenant.id, "ci")
    first = (await revoke_api_key(session, tenant.id, key.id)).revoked_at

    second = (await revoke_api_key(session, tenant.id, key.id)).revoked_at

    assert first == second


async def test_revoking_one_key_leaves_the_others_alone(
    session: AsyncSession, tenant: Tenant
) -> None:
    doomed, _ = await create_api_key(session, tenant.id, "one")
    survivor, _ = await create_api_key(session, tenant.id, "two")

    await revoke_api_key(session, tenant.id, doomed.id)

    await session.refresh(survivor)
    assert survivor.status() == "active"


async def test_a_tenant_cannot_revoke_another_tenants_key(
    session: AsyncSession, tenant: Tenant
) -> None:
    key, _ = await create_api_key(session, tenant.id, "ci")
    intruder = await other_tenant(session)

    with pytest.raises(KeyNotFoundError):
        await revoke_api_key(session, intruder.id, key.id)

    await session.refresh(key)
    assert key.status() == "active"


async def test_revoking_an_unknown_key_fails(session: AsyncSession, tenant: Tenant) -> None:
    with pytest.raises(KeyNotFoundError):
        await revoke_api_key(session, tenant.id, "nope")


async def test_revoked_keys_are_still_found_but_report_their_status(
    session: AsyncSession, tenant: Tenant
) -> None:
    key, secret = await create_api_key(session, tenant.id, "ci")
    await revoke_api_key(session, tenant.id, key.id)

    found = await find_key_by_secret(session, secret)

    assert found is not None
    assert found.status() == "revoked"


async def test_rotation_issues_a_new_key_and_revokes_the_old_one(
    session: AsyncSession, tenant: Tenant
) -> None:
    old, old_secret = await create_api_key(session, tenant.id, "ci pipeline")

    new, new_secret = await rotate_api_key(session, tenant.id, old.id)

    assert new.id != old.id
    assert new_secret != old_secret
    assert new.key_hash != old.key_hash
    assert (new.tenant_id, new.name) == (tenant.id, "ci pipeline")
    assert new.status() == "active"
    await session.refresh(old)
    assert old.status() == "revoked"


async def test_rotation_with_grace_keeps_the_old_key_alive_for_a_while(
    session: AsyncSession, tenant: Tenant
) -> None:
    old, _ = await create_api_key(session, tenant.id, "ci")

    await rotate_api_key(session, tenant.id, old.id, grace=HOUR)

    await session.refresh(old)
    assert old.status() == "active"
    assert old.revoked_at is None
    assert old.expires_at is not None
    assert abs((old.expires_at - (now() + HOUR)).total_seconds()) < 5
    assert old.status(now() + 2 * HOUR) == "expired"


async def test_grace_never_extends_an_earlier_expiry(session: AsyncSession, tenant: Tenant) -> None:
    soon = now() + timedelta(minutes=10)
    old, _ = await create_api_key(session, tenant.id, "ci", expires_at=soon)

    await rotate_api_key(session, tenant.id, old.id, grace=24 * HOUR)

    await session.refresh(old)
    assert old.expires_at == soon


async def test_new_key_does_not_inherit_the_old_expiry(
    session: AsyncSession, tenant: Tenant
) -> None:
    old, _ = await create_api_key(session, tenant.id, "ci", expires_at=now() + HOUR)

    unlimited, _ = await rotate_api_key(session, tenant.id, old.id)
    fresh_old, _ = await create_api_key(session, tenant.id, "ci2")
    limited_deadline = now() + 24 * HOUR
    limited, _ = await rotate_api_key(session, tenant.id, fresh_old.id, expires_at=limited_deadline)

    assert unlimited.expires_at is None
    assert limited.expires_at == limited_deadline


@pytest.mark.parametrize("how", ["revoked", "expired"])
async def test_dead_keys_cannot_be_rotated(session: AsyncSession, tenant: Tenant, how: str) -> None:
    key, _ = await create_api_key(session, tenant.id, "ci", expires_at=now() + HOUR)
    if how == "revoked":
        await revoke_api_key(session, tenant.id, key.id)
    else:
        await session.execute(update(ApiKey).values(expires_at=now() - HOUR))
        await session.commit()
        await session.refresh(key)

    with pytest.raises(KeyInactiveError) as excinfo:
        await rotate_api_key(session, tenant.id, key.id)

    assert excinfo.value.status == how
    count = len((await session.execute(select(ApiKey))).scalars().all())
    assert count == 1  # nothing was created


async def test_a_tenant_cannot_rotate_another_tenants_key(
    session: AsyncSession, tenant: Tenant
) -> None:
    key, _ = await create_api_key(session, tenant.id, "ci")
    intruder = await other_tenant(session)

    with pytest.raises(KeyNotFoundError):
        await rotate_api_key(session, intruder.id, key.id)

    assert len((await session.execute(select(ApiKey))).scalars().all()) == 1


async def test_negative_grace_is_rejected(session: AsyncSession, tenant: Tenant) -> None:
    key, _ = await create_api_key(session, tenant.id, "ci")

    with pytest.raises(ValueError, match="grace"):
        await rotate_api_key(session, tenant.id, key.id, grace=-HOUR)


# ---------- enforcement over HTTP ----------


def run(client: AuthedClient, app: FastAPI, work: Callable[[AsyncSession], Awaitable[T]]) -> T:
    async def call() -> T:
        async with app.state.db_session_factory() as session:
            return await work(session)

    assert client.portal is not None
    result: T = client.portal.call(call)
    return result


def own_key_id(client: AuthedClient, app: FastAPI) -> tuple[str, str]:
    async def find(session: AsyncSession) -> tuple[str, str]:
        key = (await session.execute(select(ApiKey))).scalar_one()
        return key.id, key.tenant_id

    return run(client, app, find)


def bearer(secret: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {secret}"}


def test_revoked_key_is_rejected_immediately(client: AuthedClient, app_: FastAPI) -> None:
    key_id, tenant_id = own_key_id(client, app_)
    assert client.get("/v1/models").status_code == 200

    run(client, app_, lambda s: revoke_api_key(s, tenant_id, key_id))
    response = client.get("/v1/models")

    assert response.status_code == 401
    error = response.json()["error"]
    assert error["code"] == "api_key_revoked"
    assert error["type"] == "authentication_error"
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert client.secret not in response.text


def test_expired_key_is_rejected(client: AuthedClient, app_: FastAPI) -> None:
    async def expire(session: AsyncSession) -> None:
        await session.execute(update(ApiKey).values(expires_at=now() - timedelta(seconds=1)))
        await session.commit()

    run(client, app_, expire)
    response = client.get("/v1/models")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "api_key_expired"


def test_key_that_has_not_yet_expired_still_works(client: AuthedClient, app_: FastAPI) -> None:
    async def set_future(session: AsyncSession) -> None:
        await session.execute(update(ApiKey).values(expires_at=now() + HOUR))
        await session.commit()

    run(client, app_, set_future)

    assert client.get("/v1/models").status_code == 200


def test_a_revoked_key_does_not_affect_siblings(client: AuthedClient, app_: FastAPI) -> None:
    key_id, tenant_id = own_key_id(client, app_)
    _, sibling_secret = run(client, app_, lambda s: create_api_key(s, tenant_id, "sibling"))

    run(client, app_, lambda s: revoke_api_key(s, tenant_id, key_id))

    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers=bearer(sibling_secret)).status_code == 200


def test_rejected_keys_are_not_marked_as_used(client: AuthedClient, app_: FastAPI) -> None:
    key_id, tenant_id = own_key_id(client, app_)
    run(client, app_, lambda s: revoke_api_key(s, tenant_id, key_id))

    client.get("/v1/models")

    async def last_used(session: AsyncSession) -> datetime | None:
        return (await session.execute(select(ApiKey.last_used_at))).scalar_one()

    assert run(client, app_, last_used) is None


def test_rotation_without_grace_switches_keys_at_once(client: AuthedClient, app_: FastAPI) -> None:
    key_id, tenant_id = own_key_id(client, app_)

    _, new_secret = run(client, app_, lambda s: rotate_api_key(s, tenant_id, key_id))

    assert client.get("/v1/models").status_code == 401  # the old key
    assert client.get("/v1/models", headers=bearer(new_secret)).status_code == 200


def test_rotation_with_grace_lets_both_keys_work_until_it_ends(
    client: AuthedClient, app_: FastAPI
) -> None:
    key_id, tenant_id = own_key_id(client, app_)
    _, new_secret = run(client, app_, lambda s: rotate_api_key(s, tenant_id, key_id, grace=HOUR))

    assert client.get("/v1/models").status_code == 200  # old key, inside the grace period
    assert client.get("/v1/models", headers=bearer(new_secret)).status_code == 200

    async def end_grace(session: AsyncSession) -> None:
        await session.execute(
            update(ApiKey)
            .where(ApiKey.id == key_id)
            .values(expires_at=now() - timedelta(seconds=1))
        )
        await session.commit()

    run(client, app_, end_grace)

    old = client.get("/v1/models")
    assert old.status_code == 401
    assert old.json()["error"]["code"] == "api_key_expired"
    assert client.get("/v1/models", headers=bearer(new_secret)).status_code == 200


def test_sdk_sees_a_revoked_key_as_an_authentication_error(
    client: AuthedClient, app_: FastAPI
) -> None:
    key_id, tenant_id = own_key_id(client, app_)
    run(client, app_, lambda s: revoke_api_key(s, tenant_id, key_id))
    sdk = OpenAI(base_url="http://testserver/v1", api_key=client.secret, http_client=client)

    with pytest.raises(AuthenticationError) as excinfo:
        sdk.models.list()

    assert excinfo.value.code == "api_key_revoked"


# ---------- migration ----------


def test_migration_keeps_existing_keys(tmp_path: Path) -> None:
    database = tmp_path / "upgrade.db"
    config = alembic_config(database)
    command.upgrade(config, "0001")
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
        row: Any = connection.execute(
            "SELECT name, expires_at, revoked_at FROM api_keys WHERE id = 'k1'"
        ).fetchone()
    assert row == ("old key", None, None)  # still there, and still usable


def test_migration_can_be_reversed_to_the_previous_revision(tmp_path: Path) -> None:
    database = tmp_path / "downgrade.db"
    config = alembic_config(database)
    command.upgrade(config, "head")

    command.downgrade(config, "0001")

    with sqlite3.connect(database) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(api_keys)")]
    assert "expires_at" not in columns
    assert "revoked_at" not in columns
    assert "key_hash" in columns
