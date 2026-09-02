from pathlib import Path

import pytest
from alembic import command
from fastapi import FastAPI
from sqlalchemy import select

from app.auth.bootstrap import bootstrap_admin
from app.auth.service import find_key_by_secret
from app.cli import main
from app.db.models import Tenant
from app.db.session import create_engine, create_session_factory
from tests.conftest import AuthedClient
from tests.test_db import alembic_config


def migrated_database(tmp_path: Path) -> str:
    database = tmp_path / "cli.db"
    command.upgrade(alembic_config(database), "head")
    return f"sqlite+aiosqlite:///{database}"


def printed_secret(out: str) -> str:
    return out.strip().splitlines()[-1]


async def load(url: str, secret: str) -> tuple[list[str], str]:
    engine = create_engine(url)
    try:
        async with create_session_factory(engine)() as session:
            key = await find_key_by_secret(session, secret)
            assert key is not None
            return list(key.scopes), key.tenant.name
    finally:
        await engine.dispose()


async def tenant_names(url: str) -> list[str]:
    engine = create_engine(url)
    try:
        async with create_session_factory(engine)() as session:
            return list((await session.execute(select(Tenant.name))).scalars())
    finally:
        await engine.dispose()


def test_create_admin_prints_a_working_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    url = migrated_database(tmp_path)

    code = main(["create-admin"], database_url=url)

    out = capsys.readouterr().out
    secret = printed_secret(out)
    assert code == 0
    assert secret.startswith("llmr_")
    assert "cannot be shown again" in out
    import asyncio

    scopes, tenant = asyncio.run(load(url, secret))
    assert scopes == ["admin:keys", "admin:tenants"]
    assert tenant == "platform"


def test_options_choose_the_tenant_and_label(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    url = migrated_database(tmp_path)

    main(["create-admin", "--tenant", "ops", "--name", "laptop"], database_url=url)

    secret = printed_secret(capsys.readouterr().out)
    import asyncio

    assert asyncio.run(load(url, secret))[1] == "ops"


def test_running_twice_reuses_the_tenant_but_issues_a_new_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    url = migrated_database(tmp_path)
    import asyncio

    main(["create-admin"], database_url=url)
    first = printed_secret(capsys.readouterr().out)
    main(["create-admin"], database_url=url)
    second = printed_secret(capsys.readouterr().out)

    assert first != second
    assert asyncio.run(tenant_names(url)) == ["platform"]
    assert asyncio.run(load(url, first))[1] == asyncio.run(load(url, second))[1]


def test_an_unmigrated_database_gives_a_helpful_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path}/empty.db"

    code = main(["create-admin"], database_url=url)

    captured = capsys.readouterr()
    assert code == 1
    assert "make migrate" in captured.err
    assert captured.out == ""  # nothing that looks like a key was printed


def test_a_blank_tenant_name_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    url = migrated_database(tmp_path)

    code = main(["create-admin", "--tenant", "  "], database_url=url)

    assert code == 1
    assert "error" in capsys.readouterr().err


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        main([])


# ---------- bootstrapping a running gateway ----------


def test_a_bootstrapped_key_can_run_the_admin_api(client: AuthedClient, app_: FastAPI) -> None:
    async def make() -> str:
        async with app_.state.db_session_factory() as session:
            _, _, secret = await bootstrap_admin(session, "platform-admins", "first key")
            return secret

    assert client.portal is not None
    secret = client.portal.call(make)
    headers = client.as_key(secret)

    tenant = client.post("/admin/tenants", json={"name": "customer"}, headers=headers).json()
    key = client.post(
        f"/admin/tenants/{tenant['id']}/keys", json={"name": "app"}, headers=headers
    ).json()
    models = client.get("/v1/models", headers=client.as_key(key["secret"]))

    assert models.status_code == 200
    assert client.get("/v1/models", headers=headers).status_code == 403  # admins cannot call models
