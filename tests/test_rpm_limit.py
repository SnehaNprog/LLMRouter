import asyncio
import logging
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest
from alembic import command
from fastapi import FastAPI
from openai import OpenAI, RateLimitError
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from app.auth.dependencies import AuthContext
from app.core.config import Settings
from app.core.redis import build_redis_client
from app.main import create_app
from app.providers.fake import FakeProvider
from app.quota.bucket import BucketStore, TakeResult
from app.quota.limiter import RateLimiter, RateLimitExceededError
from app.router.registry import ModelRegistry
from tests.conftest import AuthedClient, registry_with
from tests.test_db import alembic_config

NOW = 1_700_000_000_000


def auth(key_id: str = "key-1", rpm_limit: int | None = None) -> AuthContext:
    return AuthContext(
        key_id=key_id,
        key_prefix="llmr_abc",
        tenant_id="tenant-1",
        tenant_name="acme",
        scopes=frozenset({"chat:write"}),
        allowed_models=None,
        rpm_limit=rpm_limit,
    )


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> int:
        return self.now


# ---------- the limiter (fake and, opt-in, real Redis) ----------


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def limiter(redis_client: Redis, clock: Clock) -> RateLimiter:
    return RateLimiter(BucketStore(redis_client), now_ms=clock)


async def hits(limiter: RateLimiter, who: AuthContext, count: int) -> list[bool]:
    outcomes = []
    for _ in range(count):
        try:
            await limiter.check_request(who)
            outcomes.append(True)
        except RateLimitExceededError:
            outcomes.append(False)
    return outcomes


async def test_requests_up_to_the_limit_pass_then_stop(limiter: RateLimiter) -> None:
    outcomes = await hits(limiter, auth(rpm_limit=5), 8)

    assert outcomes == [True] * 5 + [False] * 3


async def test_the_error_names_the_limit_and_how_long_to_wait(limiter: RateLimiter) -> None:
    await hits(limiter, auth(rpm_limit=60), 60)

    with pytest.raises(RateLimitExceededError) as excinfo:
        await limiter.check_request(auth(rpm_limit=60))

    assert excinfo.value.limit == 60
    assert excinfo.value.retry_after == pytest.approx(1.0)  # 60 a minute is one a second
    assert "60 requests per minute" in str(excinfo.value)
    assert "1s" in str(excinfo.value)


async def test_a_slow_limit_asks_for_a_longer_wait(limiter: RateLimiter) -> None:
    await hits(limiter, auth(rpm_limit=2), 2)

    with pytest.raises(RateLimitExceededError) as excinfo:
        await limiter.check_request(auth(rpm_limit=2))

    assert excinfo.value.retry_after == pytest.approx(30.0)  # two a minute is one every 30 s


async def test_allowance_returns_gradually(limiter: RateLimiter, clock: Clock) -> None:
    who = auth(rpm_limit=60)
    await hits(limiter, who, 60)
    assert (await hits(limiter, who, 1)) == [False]

    clock.now += 1_000  # one second later: one request's worth
    after_a_second = await hits(limiter, who, 3)
    clock.now += 10_000
    after_ten_more = await hits(limiter, who, 12)

    assert after_a_second == [True, False, False]
    assert after_ten_more == [True] * 10 + [False] * 2


async def test_a_quiet_key_can_burst_to_its_full_limit(limiter: RateLimiter, clock: Clock) -> None:
    who = auth(rpm_limit=30)
    clock.now += 3_600_000  # an hour of silence

    assert await hits(limiter, who, 31) == [True] * 30 + [False]


async def test_keys_are_limited_separately(limiter: RateLimiter) -> None:
    busy, calm = auth("busy", rpm_limit=3), auth("calm", rpm_limit=3)
    await hits(limiter, busy, 3)

    assert await hits(limiter, busy, 1) == [False]
    assert await hits(limiter, calm, 3) == [True, True, True]


async def test_each_key_has_its_own_limit(limiter: RateLimiter) -> None:
    small, large = auth("small", rpm_limit=2), auth("large", rpm_limit=10)

    assert (await hits(limiter, small, 5)).count(True) == 2
    assert (await hits(limiter, large, 15)).count(True) == 10


async def test_the_default_applies_to_keys_without_a_limit(
    redis_client: Redis, clock: Clock
) -> None:
    limiter = RateLimiter(BucketStore(redis_client), default_rpm=4, now_ms=clock)

    assert (await hits(limiter, auth(rpm_limit=None), 9)).count(True) == 4


async def test_a_keys_own_limit_beats_the_default(redis_client: Redis, clock: Clock) -> None:
    limiter = RateLimiter(BucketStore(redis_client), default_rpm=4, now_ms=clock)

    assert (await hits(limiter, auth("high", rpm_limit=8), 12)).count(True) == 8
    assert (await hits(limiter, auth("low", rpm_limit=2), 12)).count(True) == 2


async def test_without_any_limit_nothing_is_counted() -> None:
    class Forbidden(BucketStore):
        def __init__(self) -> None:  # no Redis at all
            pass

        async def take(self, *args: Any, **kwargs: Any) -> TakeResult:
            raise AssertionError("Redis must not be touched when there is no limit")

    unlimited = RateLimiter(Forbidden(), default_rpm=None)

    for _ in range(50):
        await unlimited.check_request(auth(rpm_limit=None))


async def test_concurrent_requests_never_exceed_the_limit(limiter: RateLimiter) -> None:
    who = auth(rpm_limit=10)

    results = await asyncio.gather(*(hits(limiter, who, 1) for _ in range(60)))

    assert sum(r[0] for r in results) == 10


# ---------- when Redis misbehaves ----------


class BrokenStore(BucketStore):
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def take(self, *args: Any, **kwargs: Any) -> TakeResult:
        raise self._error


@pytest.mark.parametrize(
    "error", [RedisConnectionError("down"), TimeoutError("slow"), ConnectionRefusedError("no")]
)
async def test_redis_failures_do_not_block_requests(
    error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    limiter = RateLimiter(BrokenStore(error), default_rpm=1)

    with caplog.at_level(logging.WARNING, logger="app.quota.limiter"):
        await limiter.check_request(auth())
        await limiter.check_request(auth())  # still allowed, although the limit is 1

    assert any("allowing request" in r.getMessage() for r in caplog.records)


async def test_programming_errors_are_not_swallowed() -> None:
    limiter = RateLimiter(BrokenStore(ValueError("a bug")), default_rpm=1)

    with pytest.raises(ValueError, match="a bug"):
        await limiter.check_request(auth())


# ---------- over HTTP ----------

BODY = {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]}


def make_client(provider: FakeProvider | None = None, **settings: Any) -> AuthedClient:
    chosen = provider or FakeProvider()

    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return registry_with(chosen)

    options = {"retry_base_delay": 0.0, **settings}
    return AuthedClient(create_app(Settings(_env_file=None, **options), factory))


def chat(client: AuthedClient, secret: str, **extra: Any) -> httpx.Response:
    return client.post(
        "/v1/chat/completions", json={**BODY, **extra}, headers=client.as_key(secret)
    )


def test_a_key_is_stopped_at_its_limit() -> None:
    with make_client() as client:
        secret = client.issue_key(rpm_limit=3)

        statuses = [chat(client, secret).status_code for _ in range(5)]

    assert statuses == [200, 200, 200, 429, 429]


def test_the_429_is_an_openai_shaped_error_with_retry_after() -> None:
    with make_client() as client:
        secret = client.issue_key(rpm_limit=1)
        chat(client, secret)

        response = chat(client, secret)

    assert response.status_code == 429
    assert response.json()["error"] == {
        "message": "Rate limit reached: 1 requests per minute. Try again in 60s.",
        "type": "rate_limit_error",
        "param": None,
        "code": "rate_limit_exceeded",
    }
    assert response.headers["Retry-After"] == "60"


def test_our_limit_is_distinguishable_from_an_upstream_one() -> None:
    from app.providers.errors import RateLimited

    upstream = FakeProvider(fail_first=99, error_factory=lambda: RateLimited("vendor says slow"))
    with make_client(upstream, retry_max_attempts=1) as client:
        secret = client.issue_key(rpm_limit=1)
        chat(client, secret)  # reaches the vendor and is throttled there

        gateway_limit = chat(client, secret)  # refused by us

    assert gateway_limit.json()["error"]["code"] == "rate_limit_exceeded"
    assert upstream.calls == 1  # the second request never got that far


def test_refused_requests_never_reach_the_provider() -> None:
    provider = FakeProvider()
    with make_client(provider) as client:
        secret = client.issue_key(rpm_limit=2)

        for _ in range(10):
            chat(client, secret)

    assert provider.calls == 2


def test_streaming_requests_count_and_get_a_plain_429() -> None:
    with make_client() as client:
        secret = client.issue_key(rpm_limit=1)
        first = chat(client, secret, stream=True)

        second = chat(client, secret, stream=True)

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["content-type"].startswith("application/json")


def test_one_keys_limit_does_not_affect_another_key_or_tenant() -> None:
    with make_client() as client:
        busy = client.issue_key(rpm_limit=1)
        sibling = client.issue_key(rpm_limit=1)
        chat(client, busy)
        assert chat(client, busy).status_code == 429

        assert chat(client, sibling).status_code == 200  # same tenant, different key
        assert chat(client, client.secret).status_code == 200  # the unlimited default key


def test_only_chat_is_limited() -> None:
    with make_client() as client:
        secret = client.issue_key(rpm_limit=1)
        chat(client, secret)
        assert chat(client, secret).status_code == 429

        for _ in range(5):
            assert client.get("/v1/models", headers=client.as_key(secret)).status_code == 200
        assert client.get("/health").status_code == 200


def test_unauthenticated_and_forbidden_requests_do_not_use_up_the_allowance() -> None:
    with make_client() as client:
        secret = client.issue_key(rpm_limit=1)
        models_only = client.issue_key(scopes=["models:read"], rpm_limit=1)

        bad_key = [chat(client, "llmr_" + "x" * 43).status_code for _ in range(5)]
        no_scope = [chat(client, models_only).status_code for _ in range(5)]
        first_real = chat(client, secret).status_code

    assert bad_key == [401] * 5
    assert no_scope == [403] * 5  # refused for scope, never for rate
    assert first_real == 200


def test_the_gateway_default_limit_applies() -> None:
    with make_client(default_rpm_limit=2) as client:
        unlimited_by_key = client.secret  # no per-key limit, so the default governs

        statuses = [chat(client, unlimited_by_key).status_code for _ in range(4)]

    assert statuses == [200, 200, 429, 429]


def test_a_per_key_limit_overrides_the_default() -> None:
    with make_client(default_rpm_limit=2) as client:
        generous = client.issue_key(rpm_limit=5)

        statuses = [chat(client, generous).status_code for _ in range(7)]

    assert statuses.count(200) == 5


def test_the_allowance_comes_back_with_time() -> None:
    with make_client() as client:
        secret = client.issue_key(rpm_limit=2)
        app: FastAPI = client.app  # type: ignore[assignment]
        statuses = [chat(client, secret).status_code for _ in range(3)]
        assert statuses == [200, 200, 429]

        async def a_minute_passes() -> None:
            async for bucket in app.state.redis.scan_iter("llmrouter:quota:rpm:*"):
                stamp = int(await app.state.redis.hget(bucket, "ts"))
                await app.state.redis.hset(bucket, "ts", stamp - 61_000)

        assert client.portal is not None
        client.portal.call(a_minute_passes)

        assert chat(client, secret).status_code == 200


def test_the_sdk_sees_a_rate_limit_error() -> None:
    with make_client() as client:
        secret = client.issue_key(rpm_limit=1)
        sdk = OpenAI(
            base_url="http://testserver/v1", api_key=secret, http_client=client, max_retries=0
        )
        sdk.chat.completions.create(
            model="test-model", messages=[{"role": "user", "content": "hi"}]
        )

        with pytest.raises(RateLimitError) as excinfo:
            sdk.chat.completions.create(
                model="test-model", messages=[{"role": "user", "content": "hi"}]
            )

    assert excinfo.value.code == "rate_limit_exceeded"
    assert excinfo.value.response.headers["Retry-After"] == "60"


def test_requests_still_work_when_redis_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    unreachable = Settings(_env_file=None, redis_url="redis://127.0.0.1:1/0", redis_timeout=0.2)
    monkeypatch.setattr(
        "app.lifespan.build_redis_client", lambda _: build_redis_client(unreachable)
    )

    with make_client() as client:
        secret = client.issue_key(rpm_limit=1)

        statuses = [chat(client, secret).status_code for _ in range(3)]

    assert statuses == [200, 200, 200]  # fails open: the limit cannot be enforced, so it is not


# ---------- configuring limits ----------


def admin_headers(client: AuthedClient) -> dict[str, str]:
    return client.as_key(client.issue_key(["admin:tenants", "admin:keys"]))


def tenant_id(client: AuthedClient, headers: dict[str, str]) -> str:
    return str(client.post("/admin/tenants", json={"name": "globex"}, headers=headers).json()["id"])


def test_the_admin_api_sets_and_shows_the_limit() -> None:
    with make_client() as client:
        headers = admin_headers(client)
        tenant = tenant_id(client, headers)

        created = client.post(
            f"/admin/tenants/{tenant}/keys", json={"name": "app", "rpm_limit": 120}, headers=headers
        )
        default = client.post(
            f"/admin/tenants/{tenant}/keys", json={"name": "other"}, headers=headers
        )

    assert created.json()["rpm_limit"] == 120
    assert default.json()["rpm_limit"] is None


@pytest.mark.parametrize("value", [0, -5, 1_000_001, "fast"])
def test_invalid_limits_are_rejected(value: Any) -> None:
    with make_client() as client:
        headers = admin_headers(client)
        tenant = tenant_id(client, headers)

        response = client.post(
            f"/admin/tenants/{tenant}/keys",
            json={"name": "app", "rpm_limit": value},
            headers=headers,
        )

    assert response.status_code == 400


def test_a_key_created_through_the_api_is_limited() -> None:
    with make_client() as client:
        headers = admin_headers(client)
        tenant = tenant_id(client, headers)
        secret = client.post(
            f"/admin/tenants/{tenant}/keys", json={"name": "app", "rpm_limit": 2}, headers=headers
        ).json()["secret"]

        statuses = [chat(client, secret).status_code for _ in range(4)]

    assert statuses == [200, 200, 429, 429]


def test_rotation_keeps_the_limit_unless_told_otherwise() -> None:
    with make_client() as client:
        headers = admin_headers(client)
        tenant = tenant_id(client, headers)
        base = f"/admin/tenants/{tenant}/keys"
        keys = [
            client.post(base, json={"name": f"k{n}", "rpm_limit": 50}, headers=headers).json()
            for n in range(3)
        ]

        kept = client.post(f"{base}/{keys[0]['id']}/rotate", json={}, headers=headers).json()
        changed = client.post(
            f"{base}/{keys[1]['id']}/rotate", json={"rpm_limit": 10}, headers=headers
        ).json()
        reset = client.post(
            f"{base}/{keys[2]['id']}/rotate", json={"rpm_limit": None}, headers=headers
        ).json()

    assert (kept["rpm_limit"], changed["rpm_limit"], reset["rpm_limit"]) == (50, 10, None)


def test_migration_leaves_existing_keys_on_the_default(tmp_path: Path) -> None:
    database = tmp_path / "rpm.db"
    config = alembic_config(database)
    command.upgrade(config, "0004")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO tenants (id, name, is_active, created_at)"
            " VALUES ('t1', 'acme', 1, '2026-01-01')"
        )
        connection.execute(
            "INSERT INTO api_keys (id, tenant_id, name, key_prefix, key_hash, created_at, scopes)"
            " VALUES ('k1', 't1', 'old key', 'llmr_abc', 'hash', '2026-01-01', '[]')"
        )

    command.upgrade(config, "head")

    with sqlite3.connect(database) as connection:
        (stored,) = connection.execute("SELECT rpm_limit FROM api_keys WHERE id = 'k1'").fetchone()
    assert stored is None


def test_the_rate_limit_migration_can_be_reversed(tmp_path: Path) -> None:
    database = tmp_path / "rpm.db"
    config = alembic_config(database)
    command.upgrade(config, "head")

    command.downgrade(config, "0004")

    with sqlite3.connect(database) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(api_keys)")]
    assert "rpm_limit" not in columns
    assert "allowed_models" in columns


@pytest.mark.parametrize("backend", ["fake", "real"])
def test_concurrent_requests_through_the_whole_stack_respect_the_limit(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """40 requests race on 12 threads for a limit of 10; the quota must hold exactly."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    if backend == "real":
        url = os.environ.get("LLMROUTER_TEST_REDIS_URL")
        if not url:
            pytest.skip("set LLMROUTER_TEST_REDIS_URL to test against a real Redis")
        real = Settings(_env_file=None, redis_url=url)
        monkeypatch.setattr("app.lifespan.build_redis_client", lambda _: build_redis_client(real))

    with make_client() as client:
        app: FastAPI = client.app  # type: ignore[assignment]
        assert client.portal is not None
        client.portal.call(app.state.redis.flushdb)
        secret = client.issue_key(rpm_limit=10)

        with ThreadPoolExecutor(max_workers=12) as pool:
            statuses = list(pool.map(lambda _: chat(client, secret).status_code, range(40)))

        client.portal.call(app.state.redis.flushdb)

    assert set(statuses) == {200, 429}
    assert 10 <= statuses.count(200) <= 11  # exactly the limit (plus a refill, if the box is slow)
