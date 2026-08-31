import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from alembic import command
from fastapi import FastAPI
from openai import OpenAI, PermissionDeniedError
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.auth.model_access import model_allowed, normalize_allowed_models
from app.auth.service import INHERIT, create_api_key, find_key_by_secret, rotate_api_key
from app.core.config import Settings
from app.db.base import Base
from app.db.models import ApiKey, Tenant
from app.db.session import create_engine, create_session_factory
from app.main import create_app
from app.providers.fake import FakeProvider
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry
from tests.conftest import AuthedClient, model_config
from tests.test_db import alembic_config

# ---------- matching ----------


def test_no_allow_list_means_every_model() -> None:
    assert model_allowed(None, "anything")


def test_exact_names_match_exactly() -> None:
    assert model_allowed(["fast"], "fast")
    assert not model_allowed(["fast"], "fast-2")
    assert not model_allowed(["fast"], "slow")
    assert not model_allowed(["fast"], "Fast")


@pytest.mark.parametrize(
    ("pattern", "model", "expected"),
    [
        ("fast*", "fast", True),
        ("fast*", "fast-v2", True),
        ("fast*", "slow-fast", False),
        ("*-large", "gpt-large", True),
        ("*-large", "gpt-large-2", False),
        ("a*c", "abc", True),
        ("a*c", "ac", True),
        ("a*c", "ab", False),
        ("*", "whatever", True),
    ],
)
def test_star_matches_any_run_of_characters(pattern: str, model: str, expected: bool) -> None:
    assert model_allowed([pattern], model) is expected


def test_other_characters_are_taken_literally() -> None:
    assert model_allowed(["a.b"], "a.b")
    assert not model_allowed(["a.b"], "axb")
    assert model_allowed(["[x]"], "[x]")
    assert not model_allowed(["[x]"], "x")
    assert not model_allowed(["a?"], "ab")


def test_any_matching_entry_is_enough() -> None:
    assert model_allowed(["cheap", "fast*"], "fast-1")
    assert model_allowed(["cheap", "fast*"], "cheap")
    assert not model_allowed(["cheap", "fast*"], "premium")


def test_allow_list_is_normalized() -> None:
    assert normalize_allowed_models([" b ", "a", "b"]) == ["a", "b"]


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ([], "must not be empty"),
        ([""], "blank"),
        (["  "], "blank"),
        (["a b"], "whitespace"),
        (["x" * 201], "limited"),
    ],
)
def test_bad_allow_lists_are_rejected(bad: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_allowed_models(bad)


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


async def test_keys_are_unrestricted_by_default(session: AsyncSession, tenant: Tenant) -> None:
    key, _ = await create_api_key(session, tenant.id, "ci")

    assert key.allowed_models is None


async def test_allow_list_is_stored_normalized_and_round_trips(
    session: AsyncSession, tenant: Tenant
) -> None:
    _, secret = await create_api_key(
        session, tenant.id, "ci", allowed_models=["fast", "cheap", "fast"]
    )
    session.expire_all()

    found = await find_key_by_secret(session, secret)

    assert found is not None
    assert found.allowed_models == ["cheap", "fast"]


async def test_unrestricted_is_stored_as_null_not_json_null(
    session: AsyncSession, engine: AsyncEngine, tenant: Tenant
) -> None:
    await create_api_key(session, tenant.id, "ci")

    async with engine.connect() as connection:
        value = (
            await connection.exec_driver_sql("SELECT allowed_models FROM api_keys")
        ).scalar_one()

    assert value is None


async def test_invalid_allow_lists_create_nothing(session: AsyncSession, tenant: Tenant) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        await create_api_key(session, tenant.id, "ci", allowed_models=[])

    assert (await session.execute(select(ApiKey))).first() is None


async def test_rotation_keeps_the_allow_list(session: AsyncSession, tenant: Tenant) -> None:
    old, _ = await create_api_key(session, tenant.id, "ci", allowed_models=["cheap"])

    new, _ = await rotate_api_key(session, tenant.id, old.id)

    assert new.allowed_models == ["cheap"]


async def test_rotation_can_replace_or_lift_the_allow_list(
    session: AsyncSession, tenant: Tenant
) -> None:
    first, _ = await create_api_key(session, tenant.id, "a", allowed_models=["cheap"])
    second, _ = await create_api_key(session, tenant.id, "b", allowed_models=["cheap"])

    replaced, _ = await rotate_api_key(session, tenant.id, first.id, allowed_models=["premium"])
    lifted, _ = await rotate_api_key(session, tenant.id, second.id, allowed_models=None)

    assert replaced.allowed_models == ["premium"]
    assert lifted.allowed_models is None


async def test_rotating_an_unrestricted_key_stays_unrestricted(
    session: AsyncSession, tenant: Tenant
) -> None:
    old, _ = await create_api_key(session, tenant.id, "ci")

    new, _ = await rotate_api_key(session, tenant.id, old.id, allowed_models=INHERIT)

    assert new.allowed_models is None


# ---------- enforcement over HTTP ----------

BODY = {"model": "cheap", "messages": [{"role": "user", "content": "Hi"}]}


def two_model_client(
    cheap: FakeProvider | None = None,
    premium: FakeProvider | None = None,
    fallbacks: dict[str, list[str]] | None = None,
) -> AuthedClient:
    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return ModelRegistry(
            [
                Deployment.create(
                    model_config(name="cheap", model="cheap-up"), cheap or FakeProvider("cheap")
                ),
                Deployment.create(
                    model_config(name="premium", model="premium-up"),
                    premium or FakeProvider("premium"),
                ),
            ],
            fallbacks,
        )

    settings = Settings(_env_file=None, retry_base_delay=0.0, retry_max_attempts=1)
    return AuthedClient(create_app(settings, factory))


def chat(client: AuthedClient, secret: str, model: str, **extra: object) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        json={**BODY, "model": model, **extra},
        headers=client.as_key(secret),
    )


def test_unrestricted_key_can_use_every_model() -> None:
    with two_model_client() as client:
        assert chat(client, client.secret, "cheap").status_code == 200
        assert chat(client, client.secret, "premium").status_code == 200


def test_restricted_key_can_only_use_its_models() -> None:
    with two_model_client() as client:
        secret = client.issue_key(allowed_models=["cheap"])

        assert chat(client, secret, "cheap").status_code == 200
        assert chat(client, secret, "premium").status_code == 403


def test_denial_is_an_openai_shaped_403() -> None:
    with two_model_client() as client:
        secret = client.issue_key(allowed_models=["cheap"])

        response = chat(client, secret, "premium")

    assert response.json()["error"] == {
        "message": "This API key is not allowed to use the model 'premium'.",
        "type": "permission_error",
        "param": "model",
        "code": "model_not_allowed",
    }


def test_the_provider_is_never_called_for_a_denied_model() -> None:
    premium = FakeProvider("premium")
    with two_model_client(premium=premium) as client:
        secret = client.issue_key(allowed_models=["cheap"])

        chat(client, secret, "premium")

    assert premium.calls == 0


def test_a_denied_model_that_does_not_exist_looks_the_same() -> None:
    with two_model_client() as client:
        secret = client.issue_key(allowed_models=["cheap"])

        real = chat(client, secret, "premium")
        ghost = chat(client, secret, "ghost")

    assert ghost.status_code == real.status_code == 403  # not 404: existence is not revealed
    assert ghost.json()["error"]["code"] == "model_not_allowed"


def test_an_allowed_model_that_does_not_exist_is_a_404() -> None:
    with two_model_client() as client:
        secret = client.issue_key(allowed_models=["ghost"])

        assert chat(client, secret, "ghost").status_code == 404


def test_patterns_grant_matching_models() -> None:
    with two_model_client() as client:
        secret = client.issue_key(allowed_models=["pre*"])

        assert chat(client, secret, "premium").status_code == 200
        assert chat(client, secret, "cheap").status_code == 403


def test_streaming_requests_are_denied_with_a_plain_403() -> None:
    with two_model_client() as client:
        secret = client.issue_key(allowed_models=["cheap"])

        response = chat(client, secret, "premium", stream=True)

    assert response.status_code == 403
    assert response.headers["content-type"].startswith("application/json")


def test_scope_is_checked_before_the_model() -> None:
    with two_model_client() as client:
        secret = client.issue_key(scopes=["models:read"], allowed_models=["cheap"])

        response = chat(client, secret, "premium")

    assert response.json()["error"]["code"] == "insufficient_scope"


def test_the_model_list_shows_only_allowed_models() -> None:
    with two_model_client() as client:
        restricted = client.issue_key(allowed_models=["cheap"])

        everything = client.get("/v1/models", headers=client.as_key(client.secret)).json()
        limited = client.get("/v1/models", headers=client.as_key(restricted)).json()

    assert [m["id"] for m in everything["data"]] == ["cheap", "premium"]
    assert [m["id"] for m in limited["data"]] == ["cheap"]


def test_allow_list_changes_apply_to_the_next_request() -> None:
    with two_model_client() as client:
        secret = client.issue_key(allowed_models=["cheap"])
        assert chat(client, secret, "premium").status_code == 403

        async def widen() -> None:
            async with client.app.state.db_session_factory() as session:  # type: ignore[attr-defined]
                await session.execute(
                    update(ApiKey)
                    .where(ApiKey.key_prefix == secret[:12])
                    .values(allowed_models=["cheap", "premium"])
                )
                await session.commit()

        assert client.portal is not None
        client.portal.call(widen)

        assert chat(client, secret, "premium").status_code == 200


def test_fallbacks_are_the_operators_choice_and_are_not_filtered() -> None:
    """A key limited to 'cheap' still benefits from cheap's configured fallback."""
    dead = FakeProvider("cheap", fail_first=99)
    premium = FakeProvider("premium", reply="rescued by premium")
    with two_model_client(cheap=dead, premium=premium, fallbacks={"cheap": ["premium"]}) as client:
        secret = client.issue_key(allowed_models=["cheap"])

        response = chat(client, secret, "cheap")

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "rescued by premium"


def test_sdk_raises_permission_denied_for_a_forbidden_model() -> None:
    with two_model_client() as client:
        secret = client.issue_key(allowed_models=["cheap"])
        sdk = OpenAI(base_url="http://testserver/v1", api_key=secret, http_client=client)

        with pytest.raises(PermissionDeniedError) as excinfo:
            sdk.chat.completions.create(
                model="premium", messages=[{"role": "user", "content": "hi"}]
            )

    assert excinfo.value.code == "model_not_allowed"


def test_default_fixture_client_is_unaffected(client: AuthedClient, app_: FastAPI) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200


# ---------- migration ----------


def test_migration_leaves_existing_keys_unrestricted(tmp_path: Path) -> None:
    database = tmp_path / "allow.db"
    config = alembic_config(database)
    command.upgrade(config, "0003")
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
        (stored,) = connection.execute(
            "SELECT allowed_models FROM api_keys WHERE id = 'k1'"
        ).fetchone()
    assert stored is None


def test_allow_list_migration_can_be_reversed(tmp_path: Path) -> None:
    database = tmp_path / "allow.db"
    config = alembic_config(database)
    command.upgrade(config, "head")

    command.downgrade(config, "0003")

    with sqlite3.connect(database) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(api_keys)")]
    assert "allowed_models" not in columns
    assert "scopes" in columns
