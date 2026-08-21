import hashlib
import re
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.auth.keys import KEY_PREFIX, generate_key, hash_key, looks_like_key
from app.auth.service import (
    TenantInactiveError,
    TenantNotFoundError,
    create_api_key,
    find_key_by_secret,
)
from app.db.base import Base
from app.db.models import ApiKey, Tenant
from app.db.session import create_engine, create_session_factory

# ---------- key generation ----------


def test_keys_have_the_expected_shape() -> None:
    generated = generate_key()

    assert generated.secret.startswith("llmr_")
    assert len(generated.secret) == 5 + 43  # prefix + 32 random bytes, url-safe base64
    assert re.fullmatch(r"llmr_[A-Za-z0-9_-]{43}", generated.secret)


def test_keys_are_unique() -> None:
    secrets_seen = {generate_key().secret for _ in range(2_000)}

    assert len(secrets_seen) == 2_000


def test_prefix_is_the_start_of_the_secret_and_fits_the_column() -> None:
    generated = generate_key()

    assert generated.secret.startswith(generated.prefix)
    assert generated.prefix.startswith(KEY_PREFIX)
    assert len(generated.prefix) == 12 <= 16


def test_prefix_reveals_little_of_the_secret() -> None:
    generated = generate_key()

    assert len(generated.secret) - len(generated.prefix) >= 36


def test_hash_is_sha256_of_the_secret() -> None:
    generated = generate_key()

    assert generated.hash == hashlib.sha256(generated.secret.encode()).hexdigest()
    assert len(generated.hash) == 64


def test_hash_known_vector() -> None:
    assert hash_key("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_hash_does_not_contain_the_secret() -> None:
    generated = generate_key()

    assert generated.secret not in generated.hash
    assert generated.secret[5:] not in generated.hash


def test_hash_is_deterministic() -> None:
    assert hash_key("same") == hash_key("same")
    assert hash_key("same") != hash_key("Same")


@pytest.mark.parametrize(
    "value",
    ["", "llmr_", "sk-abc", "llmr_tooshort", " " + generate_key().secret, "x" * 48],
)
def test_malformed_keys_are_rejected_cheaply(value: str) -> None:
    assert not looks_like_key(value)


def test_generated_keys_pass_the_shape_check() -> None:
    assert all(looks_like_key(generate_key().secret) for _ in range(50))


# ---------- storing and finding keys ----------


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


async def test_create_returns_the_record_and_the_secret(
    session: AsyncSession, tenant: Tenant
) -> None:
    key, secret = await create_api_key(session, tenant.id, "ci pipeline")

    assert key.tenant_id == tenant.id
    assert key.name == "ci pipeline"
    assert key.key_hash == hash_key(secret)
    assert secret.startswith(key.key_prefix)
    assert looks_like_key(secret)


async def test_plaintext_secret_is_never_stored(
    session: AsyncSession, engine: AsyncEngine, tenant: Tenant
) -> None:
    _, secret = await create_api_key(session, tenant.id, "ci")

    async with engine.connect() as connection:
        rows = (await connection.exec_driver_sql("SELECT * FROM api_keys")).all()
        everything = " ".join(str(value) for row in rows for value in row)

    assert secret not in everything
    assert secret[5:] not in everything  # not even the random part
    assert hash_key(secret) in everything


async def test_secret_is_not_exposed_by_the_model(session: AsyncSession, tenant: Tenant) -> None:
    key, secret = await create_api_key(session, tenant.id, "ci")

    assert secret not in repr(key)
    assert secret not in str(key.__dict__)


async def test_find_by_secret_returns_the_key_with_its_tenant(
    session: AsyncSession, engine: AsyncEngine, tenant: Tenant
) -> None:
    key, secret = await create_api_key(session, tenant.id, "ci")

    async with create_session_factory(engine)() as fresh:  # a new session: nothing cached
        found = await find_key_by_secret(fresh, secret)

        assert found is not None
        assert found.id == key.id
        assert found.tenant.name == "acme"  # loaded eagerly, no lazy-load error


async def test_wrong_or_modified_secrets_find_nothing(
    session: AsyncSession, tenant: Tenant
) -> None:
    _, secret = await create_api_key(session, tenant.id, "ci")
    flipped = secret[:-1] + ("A" if secret[-1] != "A" else "B")

    assert await find_key_by_secret(session, flipped) is None
    assert await find_key_by_secret(session, secret.upper()) is None
    assert await find_key_by_secret(session, "not-a-key") is None
    assert await find_key_by_secret(session, "") is None
    assert await find_key_by_secret(session, generate_key().secret) is None


async def test_the_hash_cannot_be_used_as_the_key(session: AsyncSession, tenant: Tenant) -> None:
    key, _ = await create_api_key(session, tenant.id, "ci")

    assert await find_key_by_secret(session, key.key_hash) is None


async def test_each_key_is_independent(session: AsyncSession, tenant: Tenant) -> None:
    first, first_secret = await create_api_key(session, tenant.id, "one")
    second, second_secret = await create_api_key(session, tenant.id, "two")

    assert first_secret != second_secret
    found_first = await find_key_by_secret(session, first_secret)
    found_second = await find_key_by_secret(session, second_secret)
    assert found_first is not None and found_first.id == first.id
    assert found_second is not None and found_second.id == second.id


async def test_new_keys_start_unused(session: AsyncSession, tenant: Tenant) -> None:
    key, _ = await create_api_key(session, tenant.id, "ci")

    assert key.last_used_at is None


async def test_unknown_tenant_is_rejected(session: AsyncSession) -> None:
    with pytest.raises(TenantNotFoundError):
        await create_api_key(session, "no-such-tenant", "ci")

    assert (await session.execute(select(ApiKey))).first() is None


async def test_inactive_tenant_cannot_get_keys(session: AsyncSession) -> None:
    dormant = Tenant(name="dormant", is_active=False)
    session.add(dormant)
    await session.commit()

    with pytest.raises(TenantInactiveError):
        await create_api_key(session, dormant.id, "ci")


@pytest.mark.parametrize("name", ["", "   ", "x" * 201])
async def test_bad_names_are_rejected(session: AsyncSession, tenant: Tenant, name: str) -> None:
    with pytest.raises(ValueError):
        await create_api_key(session, tenant.id, name)


async def test_names_are_trimmed(session: AsyncSession, tenant: Tenant) -> None:
    key, _ = await create_api_key(session, tenant.id, "  ci  ")

    assert key.name == "ci"
