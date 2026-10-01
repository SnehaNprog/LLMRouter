import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from alembic import command
from sqlalchemy import func, select

from app.cli import main
from app.core.config import Settings
from app.db.models import RequestLog, Tenant, new_id
from app.db.session import create_engine, create_session_factory
from app.main import create_app
from app.providers.fake import FakeProvider
from app.router.registry import ModelRegistry
from app.usage.queries import decode_cursor, encode_cursor, month_start
from tests.conftest import AuthedClient, registry_with
from tests.test_db import alembic_config

D = datetime  # shorthand
OCT = D(2026, 10, 1, tzinfo=UTC)


def new_client() -> AuthedClient:
    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return registry_with(FakeProvider())

    return AuthedClient(create_app(Settings(_env_file=None, retry_base_delay=0.0), factory))


def log_row(tenant_id: str, key_id: str = "key-a", **kwargs: Any) -> RequestLog:
    created: datetime = kwargs.pop("created", OCT + timedelta(hours=1))
    fields: dict[str, Any] = {
        "id": new_id(),
        "created_at": created,
        "day": created.strftime("%Y-%m-%d"),
        "tenant_id": tenant_id,
        "key_id": key_id,
        "requested_model": "fast",
        "served_model": "fast",
        "status_code": 200,
        "stream": False,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cost_nano_usd": 1_000_000,
        "latency_ms": 100,
        "routed": False,
        "escalated": False,
    }
    fields.update(kwargs)
    return RequestLog(**fields)


def seed(client: AuthedClient, *rows: RequestLog) -> None:
    async def insert() -> None:
        async with client.app.state.db_session_factory() as session:  # type: ignore[attr-defined]
            session.add_all(rows)
            await session.commit()

    assert client.portal is not None
    client.portal.call(insert)


def my_tenant(client: AuthedClient) -> str:
    async def fetch() -> str:
        async with client.app.state.db_session_factory() as session:  # type: ignore[attr-defined]
            return str((await session.execute(select(Tenant.id))).scalars().first())

    assert client.portal is not None
    return client.portal.call(fetch)  # type: ignore[no-any-return]


def other_tenant(client: AuthedClient, name: str = "someone-else") -> str:
    async def make() -> str:
        async with client.app.state.db_session_factory() as session:  # type: ignore[attr-defined]
            tenant = Tenant(name=name)
            session.add(tenant)
            await session.commit()
            return tenant.id

    assert client.portal is not None
    return client.portal.call(make)  # type: ignore[no-any-return]


def reader(client: AuthedClient) -> dict[str, str]:
    return client.as_key(client.issue_key(["usage:read"]))


def administrator(client: AuthedClient) -> dict[str, str]:
    return client.as_key(client.issue_key(["admin:tenants"]))


def usage(client: AuthedClient, **params: Any) -> httpx.Response:
    params.setdefault("start", "2026-10-01T00:00:00Z")
    return client.get("/v1/usage", params=params, headers=reader(client))


# ---------- who may look ----------


def test_a_key_needs_the_usage_scope() -> None:
    with new_client() as client:
        assert client.get("/v1/usage").status_code == 403  # the default key cannot
        assert client.get("/v1/usage/requests").status_code == 403
        assert usage(client).status_code == 200


def test_no_key_at_all_is_refused() -> None:
    with new_client() as client:
        assert client.get("/v1/usage", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_the_admin_endpoints_need_an_admin_scope() -> None:
    with new_client() as client:
        tenant = my_tenant(client)
        assert (
            client.get(f"/admin/tenants/{tenant}/usage", headers=reader(client)).status_code == 403
        )
        assert (
            client.get(
                f"/admin/tenants/{tenant}/usage/requests", headers=reader(client)
            ).status_code
            == 403
        )
        ok = client.get(f"/admin/tenants/{tenant}/usage", headers=administrator(client))
        assert ok.status_code == 200


def test_usage_read_is_not_part_of_the_default_scopes() -> None:
    from app.auth.scopes import DEFAULT_SCOPES

    assert "usage:read" not in DEFAULT_SCOPES


# ---------- tenant isolation ----------


def test_a_tenant_sees_only_its_own_usage() -> None:
    with new_client() as client:
        mine, theirs = my_tenant(client), other_tenant(client)
        seed(client, log_row(mine), log_row(theirs, cost_nano_usd=9_000_000_000), log_row(theirs))

        body = usage(client).json()

    assert body["totals"]["requests"] == 1
    assert body["totals"]["cost_usd"] == pytest.approx(0.001)


def test_asking_for_another_tenants_key_shows_nothing_of_it() -> None:
    with new_client() as client:
        mine, theirs = my_tenant(client), other_tenant(client)
        seed(client, log_row(mine, "key-mine"), log_row(theirs, "key-theirs"))

        body = usage(client, key_id="key-theirs").json()
        listing = client.get(
            "/v1/usage/requests",
            params={"key_id": "key-theirs", "start": "2026-10-01T00:00:00Z"},
            headers=reader(client),
        ).json()

    assert body["totals"]["requests"] == 0
    assert listing["data"] == []


def test_an_administrator_can_see_any_tenant() -> None:
    with new_client() as client:
        theirs = other_tenant(client)
        seed(client, log_row(theirs), log_row(theirs))

        response = client.get(
            f"/admin/tenants/{theirs}/usage",
            params={"start": "2026-10-01T00:00:00Z"},
            headers=administrator(client),
        )

    assert response.json()["totals"]["requests"] == 2


def test_an_unknown_tenant_is_a_404_for_the_administrator() -> None:
    with new_client() as client:
        headers = administrator(client)
        assert client.get("/admin/tenants/nope/usage", headers=headers).status_code == 404
        assert client.get("/admin/tenants/nope/usage/requests", headers=headers).status_code == 404


# ---------- the figures ----------


def test_the_totals_add_up_what_was_used() -> None:
    with new_client() as client:
        tenant = my_tenant(client)
        seed(
            client,
            log_row(tenant, prompt_tokens=100, completion_tokens=50, cost_nano_usd=2_000_000_000,
                    latency_ms=200),
            log_row(tenant, prompt_tokens=10, completion_tokens=5, cost_nano_usd=500_000_000,
                    latency_ms=100),
            log_row(tenant, status_code=502, error_code="upstream_unavailable", prompt_tokens=0,
                    completion_tokens=0, cost_nano_usd=0, latency_ms=60, served_model=None),
        )  # fmt: skip

        totals = usage(client).json()["totals"]

    assert totals == {
        "requests": 3,
        "errors": 1,
        "prompt_tokens": 110,
        "completion_tokens": 55,
        "total_tokens": 165,
        "cost_usd": pytest.approx(2.5),
        "avg_latency_ms": 120.0,
    }


def test_a_stream_that_broke_after_it_began_counts_as_an_error() -> None:
    with new_client() as client:
        tenant = my_tenant(client)
        seed(client, log_row(tenant, status_code=200, error_code="upstream_unavailable"))
        assert usage(client).json()["totals"]["errors"] == 1


def test_an_empty_period_is_all_zeros_not_an_error() -> None:
    with new_client() as client:
        body = usage(client).json()

    assert body["totals"] == {
        "requests": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "total_tokens": 0, "cost_usd": 0.0, "avg_latency_ms": 0.0,
    }  # fmt: skip
    assert body["groups"] == []


def seed_variety(client: AuthedClient) -> str:
    tenant = my_tenant(client)
    seed(
        client,
        log_row(tenant, "key-a", served_model="fast", tier="slm", routed=True,
                requested_model="auto", created=OCT + timedelta(days=1, hours=2)),
        log_row(tenant, "key-a", served_model="fast", tier="slm",
                created=OCT + timedelta(days=1, hours=3)),
        log_row(tenant, "key-b", served_model="smart", tier="frontier", requested_model="smart",
                cost_nano_usd=5_000_000, created=OCT + timedelta(days=2)),
        log_row(tenant, "key-b", served_model=None, status_code=429, error_code="budget_exceeded",
                requested_model="smart", cost_nano_usd=0, created=OCT + timedelta(days=2, hours=1)),
    )  # fmt: skip
    return tenant


def groups(client: AuthedClient, by: str) -> dict[str, dict[str, Any]]:
    body = usage(client, group_by=by).json()
    assert body["group_by"] == by
    return {g["key"]: g for g in body["groups"]}


def test_grouping_by_the_model_that_served() -> None:
    with new_client() as client:
        seed_variety(client)
        found = groups(client, "model")

    assert set(found) == {"fast", "smart", "(none)"}
    assert found["fast"]["requests"] == 2 and found["fast"]["cost_usd"] == pytest.approx(0.002)
    assert found["(none)"]["errors"] == 1


def test_grouping_by_the_model_asked_for() -> None:
    with new_client() as client:
        seed_variety(client)
        found = groups(client, "requested_model")
    assert {k: v["requests"] for k, v in found.items()} == {"auto": 1, "fast": 1, "smart": 2}


def test_grouping_by_key_tier_status_and_day() -> None:
    with new_client() as client:
        seed_variety(client)
        assert {k: v["requests"] for k, v in groups(client, "key").items()} == {
            "key-a": 2,
            "key-b": 2,
        }
        assert {k: v["requests"] for k, v in groups(client, "tier").items()} == {
            "slm": 2,
            "frontier": 1,
            "(none)": 1,
        }
        assert {k: v["requests"] for k, v in groups(client, "status").items()} == {
            "200": 3,
            "429": 1,
        }
        by_day = usage(client, group_by="day").json()["groups"]

    assert [(g["key"], g["requests"]) for g in by_day] == [("2026-10-02", 2), ("2026-10-03", 2)]


def test_busiest_groups_come_first() -> None:
    with new_client() as client:
        seed_variety(client)
        keys = [g["key"] for g in usage(client, group_by="model").json()["groups"]]
    assert keys[0] == "fast"  # two requests, against one each


def test_without_a_grouping_only_the_totals_are_given() -> None:
    with new_client() as client:
        seed_variety(client)
        body = usage(client, group_by="none").json()
    assert body["groups"] == [] and body["totals"]["requests"] == 4


def test_the_group_figures_add_up_to_the_totals() -> None:
    with new_client() as client:
        seed_variety(client)
        body = usage(client, group_by="model").json()
    for field in ("requests", "errors", "prompt_tokens", "completion_tokens"):
        assert sum(g[field] for g in body["groups"]) == body["totals"][field]
    assert sum(g["cost_usd"] for g in body["groups"]) == pytest.approx(body["totals"]["cost_usd"])


def test_a_key_can_be_singled_out() -> None:
    with new_client() as client:
        seed_variety(client)
        body = usage(client, key_id="key-a").json()
    assert body["totals"]["requests"] == 2


def test_an_unknown_grouping_is_refused() -> None:
    with new_client() as client:
        assert usage(client, group_by="colour").status_code == 400


# ---------- the period ----------


def test_the_period_is_start_inclusive_and_end_exclusive() -> None:
    with new_client() as client:
        tenant = my_tenant(client)
        edge = OCT + timedelta(days=10)
        seed(
            client,
            log_row(tenant, created=edge - timedelta(seconds=1)),
            log_row(tenant, created=edge),
            log_row(tenant, created=edge + timedelta(days=1)),
        )

        body = usage(
            client, start=edge.isoformat(), end=(edge + timedelta(days=1)).isoformat()
        ).json()

    assert body["totals"]["requests"] == 1  # only the one exactly at the start


def test_an_unzoned_time_is_taken_as_utc() -> None:
    with new_client() as client:
        tenant = my_tenant(client)
        seed(client, log_row(tenant, created=OCT + timedelta(hours=5)))
        inside = usage(client, start="2026-10-01T05:00:00").json()["totals"]["requests"]
        after = usage(client, start="2026-10-01T05:00:01").json()["totals"]["requests"]
    assert (inside, after) == (1, 0)


def test_an_end_before_the_start_is_refused() -> None:
    with new_client() as client:
        response = usage(client, start="2026-10-05T00:00:00Z", end="2026-10-04T00:00:00Z")
    assert response.status_code == 400 and "end must be after start" in response.text


def test_the_period_starts_this_month_by_default() -> None:
    with new_client() as client:
        tenant = my_tenant(client)
        now = datetime.now(UTC)
        seed(
            client,
            log_row(tenant, created=now - timedelta(seconds=5)),
            log_row(tenant, created=now - timedelta(days=40)),  # last month or earlier
        )

        body = client.get("/v1/usage", headers=reader(client)).json()

    assert body["totals"]["requests"] == 1
    assert body["start"].startswith(month_start().strftime("%Y-%m-01"))
    assert body["end"] is None


def test_the_month_starts_at_midnight_on_the_first() -> None:
    assert month_start(D(2026, 11, 17, 13, 45, tzinfo=UTC)) == D(2026, 11, 1, tzinfo=UTC)


# ---------- the request list ----------


def listing(client: AuthedClient, **params: Any) -> dict[str, Any]:
    params.setdefault("start", "2026-10-01T00:00:00Z")
    response = client.get("/v1/usage/requests", params=params, headers=reader(client))
    assert response.status_code == 200, response.text
    return response.json()  # type: ignore[no-any-return]


def test_requests_are_listed_newest_first() -> None:
    with new_client() as client:
        tenant = my_tenant(client)
        seed(client, *[log_row(tenant, created=OCT + timedelta(minutes=n)) for n in range(5)])
        stamps = [r["created_at"] for r in listing(client)["data"]]
    assert stamps == sorted(stamps, reverse=True) and len(stamps) == 5


def test_a_listed_request_shows_what_happened_and_nothing_of_its_content() -> None:
    with new_client() as client:
        tenant = my_tenant(client)
        seed(client, log_row(tenant, request_id="req-1", tier="slm", routed=True, complexity=0.2,
                             cost_nano_usd=1_500_000_000))  # fmt: skip
        [row] = listing(client)["data"]

    assert row["request_id"] == "req-1" and row["cost_usd"] == pytest.approx(1.5)
    assert row["routed"] is True and row["tier"] == "slm" and row["complexity"] == 0.2
    assert set(row) == {
        "id", "created_at", "request_id", "key_id", "requested_model", "served_model",
        "status_code", "error_code", "stream", "prompt_tokens", "completion_tokens", "cost_usd",
        "latency_ms", "routed", "tier", "escalated", "complexity",
    }  # fmt: skip


def test_paging_visits_every_request_exactly_once() -> None:
    with new_client() as client:
        tenant = my_tenant(client)
        seed(client, *[log_row(tenant, created=OCT + timedelta(minutes=n)) for n in range(23)])

        seen, cursor, pages = [], None, 0
        while True:
            page = listing(client, limit=5, **({"before": cursor} if cursor else {}))
            seen += [r["id"] for r in page["data"]]
            pages += 1
            cursor = page["next_before"]
            if cursor is None:
                break

    assert len(seen) == len(set(seen)) == 23
    assert pages == 5


def test_requests_with_the_same_timestamp_are_not_lost_between_pages() -> None:
    with new_client() as client:
        tenant = my_tenant(client)
        same = OCT + timedelta(hours=1)
        seed(client, *[log_row(tenant, created=same) for _ in range(12)])

        seen, cursor = [], None
        while True:
            page = listing(client, limit=5, **({"before": cursor} if cursor else {}))
            seen += [r["id"] for r in page["data"]]
            cursor = page["next_before"]
            if cursor is None:
                break

    assert len(seen) == len(set(seen)) == 12


def test_the_last_page_has_no_cursor() -> None:
    with new_client() as client:
        tenant = my_tenant(client)
        seed(client, *[log_row(tenant) for _ in range(5)])
        assert listing(client, limit=5)["next_before"] is None
        assert listing(client, limit=4)["next_before"] is not None


def test_the_list_can_be_filtered() -> None:
    with new_client() as client:
        seed_variety(client)
        assert len(listing(client, key_id="key-b")["data"]) == 2
        assert len(listing(client, model="smart")["data"]) == 2  # asked for or served
        assert len(listing(client, model="auto")["data"]) == 1
        failed = listing(client, errors="true")["data"]
    assert [r["error_code"] for r in failed] == ["budget_exceeded"]


@pytest.mark.parametrize("limit", [0, 201, -1])
def test_the_page_size_is_bounded(limit: int) -> None:
    with new_client() as client:
        response = client.get("/v1/usage/requests", params={"limit": limit}, headers=reader(client))
    assert response.status_code == 400


@pytest.mark.parametrize("cursor", ["garbage", "2026-10-01T00:00:00Z", "~abc", "notadate~abc"])
def test_a_forged_cursor_is_refused(cursor: str) -> None:
    with new_client() as client:
        response = client.get(
            "/v1/usage/requests", params={"before": cursor}, headers=reader(client)
        )
    assert response.status_code == 400 and "cursor" in response.text


def test_a_cursor_round_trips() -> None:
    from app.schemas.usage import RequestLogOut

    row = RequestLogOut.of(log_row("t", created=OCT + timedelta(seconds=3)))
    stamp, row_id = decode_cursor(encode_cursor(row))
    assert (stamp, row_id) == (row.created_at, row.id)


def test_the_admin_can_list_a_tenants_requests() -> None:
    with new_client() as client:
        theirs = other_tenant(client)
        seed(client, log_row(theirs), log_row(theirs))
        response = client.get(
            f"/admin/tenants/{theirs}/usage/requests",
            params={"start": "2026-10-01T00:00:00Z"},
            headers=administrator(client),
        )
    assert len(response.json()["data"]) == 2


# ---------- real traffic shows up in the report ----------


def test_requests_made_through_the_gateway_appear_in_the_usage_report() -> None:
    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return registry_with(
            FakeProvider(prompt_tokens=100, completion_tokens=50),
            input_cost_per_mtok=10.0,
            output_cost_per_mtok=30.0,
        )

    body = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
    with AuthedClient(
        create_app(Settings(_env_file=None, retry_base_delay=0.0), factory)
    ) as client:
        for _ in range(3):
            assert client.post("/v1/chat/completions", json=body).status_code == 200
        client.post("/v1/chat/completions", json={**body, "model": "nope"})
        report = client.get("/v1/usage", headers=reader(client)).json()

    assert report["totals"]["requests"] == 4 and report["totals"]["errors"] == 1
    assert report["totals"]["prompt_tokens"] == 300 and report["totals"]["completion_tokens"] == 150
    assert report["totals"]["cost_usd"] == pytest.approx(3 * (100 * 10 + 50 * 30) / 1e6)
    [served] = [g for g in report["groups"] if g["key"] == "test-model"]
    assert served["requests"] == 3


# ---------- pruning old logs ----------


def migrated(tmp_path: Path) -> str:
    database = tmp_path / "prune.db"
    command.upgrade(alembic_config(database), "head")
    return f"sqlite+aiosqlite:///{database}"


async def put_logs(url: str, ages_in_days: list[int]) -> None:
    engine = create_engine(url)
    try:
        async with create_session_factory(engine)() as session:
            tenant = Tenant(name="t")
            session.add(tenant)
            await session.commit()
            now = datetime.now(UTC)
            session.add_all(
                log_row(tenant.id, created=now - timedelta(days=age)) for age in ages_in_days
            )
            await session.commit()
    finally:
        await engine.dispose()


async def count_logs(url: str) -> int:
    engine = create_engine(url)
    try:
        async with create_session_factory(engine)() as session:
            return int(
                (await session.execute(select(func.count()).select_from(RequestLog))).scalar_one()
            )
    finally:
        await engine.dispose()


async def remaining_ages(url: str) -> list[int]:
    engine = create_engine(url)
    try:
        async with create_session_factory(engine)() as session:
            stamps = (await session.execute(select(RequestLog.created_at))).scalars().all()
            return sorted(round((datetime.now(UTC) - s).total_seconds() / 86_400) for s in stamps)
    finally:
        await engine.dispose()


def test_pruning_deletes_only_what_is_older_than_the_limit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    url = migrated(tmp_path)
    asyncio.run(put_logs(url, [1, 10, 29, 31, 60, 400]))

    code = main(["prune-logs", "--days", "30"], database_url=url)

    assert code == 0
    assert asyncio.run(remaining_ages(url)) == [1, 10, 29]  # the recent ones are the ones kept
    assert "Deleted 3 request log(s) older than 30 day(s)." in capsys.readouterr().out


def test_pruning_nothing_is_fine(tmp_path: Path) -> None:
    url = migrated(tmp_path)
    assert main(["prune-logs", "--days", "30"], database_url=url) == 0


@pytest.mark.parametrize("days", ["0", "-5"])
def test_a_retention_below_a_day_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], days: str
) -> None:
    url = migrated(tmp_path)
    asyncio.run(put_logs(url, [1000]))
    assert main(["prune-logs", "--days", days], database_url=url) == 2
    assert asyncio.run(count_logs(url)) == 1  # nothing was deleted
    assert "at least 1" in capsys.readouterr().err


def test_pruning_an_unmigrated_database_says_how_to_fix_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'empty.db'}"
    assert main(["prune-logs", "--days", "30"], database_url=url) == 1
    assert "make migrate" in capsys.readouterr().err
