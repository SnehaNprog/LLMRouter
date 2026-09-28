import asyncio
import logging
import sqlite3
from datetime import UTC, datetime
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
from app.quota.budget import BudgetStore, SpendResult, nano_to_usd, usd_to_nano
from app.quota.budget_limiter import BudgetExceededError, BudgetLimiter
from app.quota.reservation import NO_RESERVATION
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry
from app.schemas.chat import ChatCompletionRequest
from tests.conftest import AuthedClient, model_config
from tests.test_db import alembic_config

# Prices used throughout: $100 per million input tokens and $200 per million output tokens.
# A reply that uses 10 prompt and 5 completion tokens therefore costs $0.002.
PRICES: dict[str, Any] = {"input_cost_per_mtok": 100.0, "output_cost_per_mtok": 200.0}
REQUEST_COST = usd_to_nano(0.002)  # 10 * 100 + 5 * 200 = 2,000 millionths of a dollar


def auth(
    key_budget: float | None = None,
    tenant_budget: float | None = None,
    key_id: str = "key-1",
    tenant_id: str = "tenant-1",
) -> AuthContext:
    return AuthContext(
        key_id=key_id,
        key_prefix="llmr_abc",
        tenant_id=tenant_id,
        tenant_name="acme",
        scopes=frozenset({"chat:write"}),
        allowed_models=None,
        rpm_limit=None,
        tpm_limit=None,
        key_budget_nano=None if key_budget is None else usd_to_nano(key_budget),
        tenant_budget_nano=None if tenant_budget is None else usd_to_nano(tenant_budget),
        pinned_model=None,
        max_tier=None,
    )


def registry(**config: Any) -> ModelRegistry:
    return ModelRegistry([Deployment.create(model_config(**{**PRICES, **config}), FakeProvider())])


def request(max_tokens: int | None = 5, **extra: Any) -> ChatCompletionRequest:
    body: dict[str, Any] = {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return ChatCompletionRequest.model_validate({**body, **extra})


# "Hi" is 8 prompt tokens. With max_tokens=5 the worst case is 8 * $100/M + 5 * $200/M.
RESERVED = usd_to_nano((8 * 100 + 5 * 200) / 1_000_000)  # $0.0018


def room_for(reservations: float) -> float:
    """A budget in dollars that fits exactly this many worst-case reservations."""
    return nano_to_usd(round(RESERVED * reservations))


class Clock:
    def __init__(self, moment: datetime) -> None:
        self.moment = moment

    def __call__(self) -> datetime:
        return self.moment


@pytest.fixture
def clock() -> Clock:
    return Clock(datetime(2026, 11, 10, 12, 0, tzinfo=UTC))


@pytest.fixture
def store(redis_client: Redis) -> BudgetStore:
    return BudgetStore(redis_client)


@pytest.fixture
def limiter(store: BudgetStore, clock: Clock) -> BudgetLimiter:
    return BudgetLimiter(store, completion_reserve=512, clock=clock)


# ---------- estimating the cost ----------


def test_the_estimate_prices_the_prompt_and_the_completion(limiter: BudgetLimiter) -> None:
    assert limiter.estimate_cost(registry(), request(max_tokens=5), 8) == RESERVED


def test_without_max_tokens_the_default_completion_is_priced(limiter: BudgetLimiter) -> None:
    expected = usd_to_nano((8 * 100 + 512 * 200) / 1_000_000)

    assert limiter.estimate_cost(registry(), request(max_tokens=None), 8) == expected


def test_several_choices_multiply_the_completion(limiter: BudgetLimiter) -> None:
    one = limiter.estimate_cost(registry(), request(max_tokens=100), 8)
    three = limiter.estimate_cost(registry(), request(max_tokens=100, n=3), 8)

    assert three == usd_to_nano((8 * 100 + 300 * 200) / 1_000_000)
    assert three > one


def test_the_estimate_assumes_the_dearest_model_in_the_fallback_chain(
    limiter: BudgetLimiter,
) -> None:
    chain = ModelRegistry(
        [
            Deployment.create(
                model_config(
                    name="cheap", model="c", input_cost_per_mtok=1, output_cost_per_mtok=2
                ),
                FakeProvider(),
            ),
            Deployment.create(
                model_config(
                    name="premium", model="p", input_cost_per_mtok=50, output_cost_per_mtok=60
                ),
                FakeProvider(),
            ),
        ],
        {"cheap": ["premium"]},
    )
    cheap_only = request().model_copy(update={"model": "premium"})
    with_fallback = request().model_copy(update={"model": "cheap"})

    assert limiter.estimate_cost(chain, with_fallback, 8) == limiter.estimate_cost(
        chain, cheap_only, 8
    )  # "cheap" may fall back to "premium", so it is priced as premium


def test_free_models_cost_nothing(limiter: BudgetLimiter) -> None:
    free = registry(input_cost_per_mtok=0.0, output_cost_per_mtok=0.0)

    assert limiter.estimate_cost(free, request(), 8) == 0


# ---------- reserving and settling ----------


async def test_no_budget_means_no_reservation_and_no_redis(clock: Clock) -> None:
    class Forbidden(BudgetStore):
        def __init__(self) -> None:
            pass

        async def spend(self, *args: Any, **kwargs: Any) -> SpendResult:
            raise AssertionError("Redis must not be touched when nothing is capped")

    reservation = await BudgetLimiter(Forbidden(), clock=clock).reserve(
        auth(), request(), registry(), 8
    )

    assert reservation is NO_RESERVATION


async def test_free_models_reserve_nothing_even_under_a_budget(
    limiter: BudgetLimiter, store: BudgetStore
) -> None:
    free = registry(input_cost_per_mtok=0.0, output_cost_per_mtok=0.0)

    reservation = await limiter.reserve(auth(key_budget=1.0), request(), free, 8)

    assert reservation.reserved == 0
    assert await store.spent("key:key-1:2026-11") == 0  # nothing was charged
    # ... but it is a real reservation, so a move to a priced model can still be paid for.
    assert await reservation.ensure(RESERVED) is True
    assert await store.spent("key:key-1:2026-11") == RESERVED


async def test_a_reservation_is_charged_up_front(
    limiter: BudgetLimiter, store: BudgetStore
) -> None:
    reservation = await limiter.reserve(auth(key_budget=1.0), request(), registry(), 8)

    assert reservation.reserved == RESERVED
    assert await store.spent("key:key-1:2026-11") == RESERVED


async def test_settling_replaces_the_estimate_with_the_real_cost(
    limiter: BudgetLimiter, store: BudgetStore
) -> None:
    reservation = await limiter.reserve(auth(key_budget=1.0), request(), registry(), 8)

    await reservation.settle(REQUEST_COST)

    assert await store.spent("key:key-1:2026-11") == REQUEST_COST


async def test_a_failed_request_costs_nothing(limiter: BudgetLimiter, store: BudgetStore) -> None:
    reservation = await limiter.reserve(auth(key_budget=1.0), request(), registry(), 8)

    await reservation.release()

    assert await store.spent("key:key-1:2026-11") == 0


async def test_an_overrun_is_charged_in_full(limiter: BudgetLimiter, store: BudgetStore) -> None:
    reservation = await limiter.reserve(auth(key_budget=1.0), request(), registry(), 8)

    await reservation.settle(RESERVED * 3)  # the model said far more than max_tokens allowed for

    assert await store.spent("key:key-1:2026-11") == RESERVED * 3


async def test_the_budget_is_refused_once_it_cannot_cover_the_next_request(
    limiter: BudgetLimiter,
) -> None:
    capped = auth(key_budget=room_for(2))

    for _ in range(2):
        await limiter.reserve(capped, request(), registry(), 8)

    with pytest.raises(BudgetExceededError) as excinfo:
        await limiter.reserve(capped, request(), registry(), 8)

    assert excinfo.value.scope == "key"


async def test_a_request_bigger_than_the_whole_budget_is_refused(limiter: BudgetLimiter) -> None:
    with pytest.raises(BudgetExceededError):
        await limiter.reserve(auth(key_budget=0.000001), request(), registry(), 8)


# ---------- key and tenant budgets together ----------


async def test_the_request_is_charged_to_both_budgets(
    limiter: BudgetLimiter, store: BudgetStore
) -> None:
    await limiter.reserve(auth(key_budget=1.0, tenant_budget=10.0), request(), registry(), 8)

    assert await store.spent("key:key-1:2026-11") == RESERVED
    assert await store.spent("tenant:tenant-1:2026-11") == RESERVED


async def test_the_tenant_budget_is_shared_by_all_its_keys(
    limiter: BudgetLimiter, store: BudgetStore
) -> None:
    shared = room_for(2)
    first = auth(tenant_budget=shared, key_id="a")
    second = auth(tenant_budget=shared, key_id="b")

    await limiter.reserve(first, request(), registry(), 8)
    await limiter.reserve(second, request(), registry(), 8)

    with pytest.raises(BudgetExceededError) as excinfo:
        await limiter.reserve(auth(tenant_budget=shared, key_id="c"), request(), registry(), 8)
    assert excinfo.value.scope == "tenant"
    assert "organisation" in str(excinfo.value)


async def test_a_key_hitting_its_own_cap_does_not_charge_the_tenant(
    limiter: BudgetLimiter, store: BudgetStore
) -> None:
    capped = auth(key_budget=0.000001, tenant_budget=10.0)

    with pytest.raises(BudgetExceededError) as excinfo:
        await limiter.reserve(capped, request(), registry(), 8)

    assert excinfo.value.scope == "key"
    assert await store.spent("tenant:tenant-1:2026-11") == 0  # all or nothing


async def test_a_tenant_cap_refusal_does_not_charge_the_key(
    limiter: BudgetLimiter, store: BudgetStore
) -> None:
    capped = auth(key_budget=10.0, tenant_budget=0.000001)

    with pytest.raises(BudgetExceededError) as excinfo:
        await limiter.reserve(capped, request(), registry(), 8)

    assert excinfo.value.scope == "tenant"
    assert await store.spent("key:key-1:2026-11") == 0


async def test_tenants_are_isolated_from_each_other(limiter: BudgetLimiter) -> None:
    tight = room_for(1)
    await limiter.reserve(auth(tenant_budget=tight, tenant_id="acme"), request(), registry(), 8)

    await limiter.reserve(auth(tenant_budget=tight, tenant_id="globex"), request(), registry(), 8)


async def test_concurrent_requests_cannot_overspend_either_budget(
    limiter: BudgetLimiter, store: BudgetStore
) -> None:
    both = auth(key_budget=room_for(10), tenant_budget=room_for(5))

    async def attempt() -> bool:
        try:
            await limiter.reserve(both, request(), registry(), 8)
        except BudgetExceededError:
            return False
        return True

    outcomes = await asyncio.gather(*(attempt() for _ in range(60)))

    assert sum(outcomes) == 5
    assert await store.spent("key:key-1:2026-11") == 5 * RESERVED
    assert await store.spent("tenant:tenant-1:2026-11") == 5 * RESERVED


# ---------- months ----------


async def test_a_new_month_starts_with_a_fresh_budget(limiter: BudgetLimiter, clock: Clock) -> None:
    capped = auth(key_budget=room_for(1))
    await limiter.reserve(capped, request(), registry(), 8)
    with pytest.raises(BudgetExceededError):
        await limiter.reserve(capped, request(), registry(), 8)

    clock.moment = datetime(2026, 12, 1, 0, 0, 1, tzinfo=UTC)

    await limiter.reserve(capped, request(), registry(), 8)  # December's budget is untouched


async def test_the_error_says_when_the_budget_resets(limiter: BudgetLimiter, clock: Clock) -> None:
    capped = auth(key_budget=room_for(1))
    clock.moment = datetime(2026, 11, 30, 18, 0, tzinfo=UTC)
    await limiter.reserve(capped, request(), registry(), 8)

    with pytest.raises(BudgetExceededError) as excinfo:
        await limiter.reserve(capped, request(), registry(), 8)

    assert excinfo.value.retry_after == 6 * 3600
    assert "$0.0018" in str(excinfo.value)
    assert "next month" in str(excinfo.value)


async def test_counters_outlive_their_month_only_briefly(
    limiter: BudgetLimiter, redis_client: Redis, clock: Clock
) -> None:
    clock.moment = datetime(2026, 11, 30, 23, 0, tzinfo=UTC)

    await limiter.reserve(auth(key_budget=1.0), request(), registry(), 8)

    ttl = await redis_client.pttl("llmrouter:budget:key:key-1:2026-11")
    assert 0 < ttl <= (3600 + 86_400) * 1000


# ---------- when Redis misbehaves ----------


class BrokenStore(BudgetStore):
    def __init__(self, error: Exception, fail_adjust_only: bool = False) -> None:
        self._error = error
        self._fail_adjust_only = fail_adjust_only

    async def spend(self, *args: Any, **kwargs: Any) -> SpendResult:
        if self._fail_adjust_only:
            return SpendResult(True, None, {})
        raise self._error

    async def adjust(self, *args: Any, **kwargs: Any) -> dict[str, int]:
        raise self._error


@pytest.mark.parametrize("error", [RedisConnectionError("down"), TimeoutError("slow")])
async def test_reserving_fails_open(
    error: Exception, clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    limiter = BudgetLimiter(BrokenStore(error), clock=clock)

    with caplog.at_level(logging.WARNING, logger="app.quota.budget_limiter"):
        reservation = await limiter.reserve(auth(key_budget=1.0), request(), registry(), 8)

    assert reservation is NO_RESERVATION
    assert any("budget limiting unavailable" in r.getMessage() for r in caplog.records)


async def test_a_failed_settlement_is_logged_not_raised(
    clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    limiter = BudgetLimiter(BrokenStore(RedisConnectionError("down"), True), clock=clock)
    reservation = await limiter.reserve(auth(key_budget=1.0), request(), registry(), 8)

    with caplog.at_level(logging.WARNING, logger="app.quota.budget_limiter"):
        await reservation.settle(1)

    assert any("could not settle" in r.getMessage() for r in caplog.records)


# ---------- over HTTP ----------

BODY = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 5,
}


def make_client(provider: FakeProvider | None = None, **settings: Any) -> AuthedClient:
    chosen = provider or FakeProvider()  # each reply uses 10 prompt + 5 completion tokens

    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return ModelRegistry([Deployment.create(model_config(**PRICES), chosen)])

    options = {"retry_base_delay": 0.0, **settings}
    return AuthedClient(create_app(Settings(_env_file=None, **options), factory))


def chat(client: AuthedClient, secret: str, **extra: Any) -> httpx.Response:
    return client.post(
        "/v1/chat/completions", json={**BODY, **extra}, headers=client.as_key(secret)
    )


def admin(client: AuthedClient) -> dict[str, str]:
    return client.as_key(client.issue_key(["admin:tenants", "admin:keys"]))


def make_key(client: AuthedClient, **body: Any) -> str:
    """A key created through the admin API (which is how budgets are set)."""
    headers = admin(client)
    tenants = client.get("/admin/tenants", headers=headers).json()["data"]
    tenant = next(t for t in tenants if t["name"] == "test-tenant")["id"]
    response = client.post(
        f"/admin/tenants/{tenant}/keys", json={"name": "budgeted", **body}, headers=headers
    )
    assert response.status_code == 201, response.text
    return str(response.json()["secret"])


def test_a_key_stops_when_its_budget_is_spent() -> None:
    with make_client() as client:
        secret = make_key(client, monthly_budget_usd=0.01)  # five requests at $0.002 each

        statuses = [chat(client, secret).status_code for _ in range(7)]

    assert statuses == [200] * 5 + [429] * 2


def test_the_429_explains_itself_and_says_when_it_resets() -> None:
    with make_client() as client:
        secret = make_key(client, monthly_budget_usd=0.002)
        chat(client, secret)

        response = chat(client, secret)

    assert response.status_code == 429
    error = response.json()["error"]
    assert error["type"] == "insufficient_quota"
    assert error["code"] == "budget_exceeded"
    assert "monthly budget of $0.002" in error["message"]
    assert "API key" in error["message"]
    assert int(response.headers["Retry-After"]) > 0


def test_refused_requests_never_reach_the_provider() -> None:
    provider = FakeProvider()
    with make_client(provider) as client:
        secret = make_key(client, monthly_budget_usd=0.004)  # two requests

        for _ in range(6):
            chat(client, secret)

    assert provider.calls == 2


def test_the_real_cost_is_charged_not_the_estimate() -> None:
    cheap = FakeProvider(prompt_tokens=1, completion_tokens=1)  # really $0.0003 a request
    with make_client(cheap) as client:
        secret = make_key(client, monthly_budget_usd=0.0054)  # three worst cases ($0.0018 each)

        statuses = [chat(client, secret).status_code for _ in range(20)]

    # Each request needs $0.0018 of headroom but only spends $0.0003, so far more than three
    # fit: the 13th is the last that still has headroom ($0.0036 spent + $0.0018 = $0.0054).
    assert statuses.count(200) == 13
    assert statuses[:13] == [200] * 13


def test_a_pricey_reply_uses_up_more_budget() -> None:
    pricey = FakeProvider(prompt_tokens=10, completion_tokens=45)  # $0.01 a request
    with make_client(pricey) as client:
        secret = make_key(client, monthly_budget_usd=0.02)

        statuses = [chat(client, secret).status_code for _ in range(4)]

    assert statuses == [200, 200, 429, 429]


def test_keys_are_budgeted_separately() -> None:
    with make_client() as client:
        spent = make_key(client, monthly_budget_usd=0.002)
        fresh = make_key(client, monthly_budget_usd=0.002)
        chat(client, spent)
        assert chat(client, spent).status_code == 429

        assert chat(client, fresh).status_code == 200
        assert chat(client, client.secret).status_code == 200  # a key with no cap


def test_the_tenant_budget_is_shared_across_its_keys() -> None:
    with make_client() as client:
        headers = admin(client)
        tenant = next(
            t["id"]
            for t in client.get("/admin/tenants", headers=headers).json()["data"]
            if t["name"] == "test-tenant"
        )
        client.patch(
            f"/admin/tenants/{tenant}", json={"monthly_budget_usd": 0.006}, headers=headers
        )
        first, second = make_key(client), make_key(client)

        statuses = [
            chat(client, secret).status_code for secret in (first, second, first, second, first)
        ]

    assert statuses == [200, 200, 200, 429, 429]  # three requests of $0.002 fit in $0.006


def test_the_tenant_error_names_the_organisation() -> None:
    with make_client() as client:
        headers = admin(client)
        tenant = client.get("/admin/tenants", headers=headers).json()["data"][0]["id"]
        client.patch(
            f"/admin/tenants/{tenant}", json={"monthly_budget_usd": 0.001}, headers=headers
        )

        response = chat(client, client.secret)

    assert response.status_code == 429
    assert "organisation" in response.json()["error"]["message"]


def test_other_tenants_are_unaffected_by_a_spent_budget() -> None:
    with make_client() as client:
        headers = admin(client)
        tenant = client.get("/admin/tenants", headers=headers).json()["data"][0]["id"]
        client.patch(
            f"/admin/tenants/{tenant}", json={"monthly_budget_usd": 0.001}, headers=headers
        )
        other = client.post("/admin/tenants", json={"name": "globex"}, headers=headers).json()["id"]
        secret = client.post(
            f"/admin/tenants/{other}/keys", json={"name": "k"}, headers=headers
        ).json()["secret"]

        assert chat(client, client.secret).status_code == 429
        assert chat(client, secret).status_code == 200


def test_failed_requests_cost_nothing() -> None:
    provider = FakeProvider(fail_first=99)
    with make_client(provider, retry_max_attempts=1) as client:
        secret = make_key(client, monthly_budget_usd=0.003)  # not even two worst cases

        failures = [chat(client, secret).status_code for _ in range(10)]
        provider.fail_first = provider.calls
        recovered = chat(client, secret).status_code

    assert failures == [502] * 10
    assert recovered == 200


def test_requests_for_unknown_models_cost_nothing() -> None:
    with make_client() as client:
        secret = make_key(client, monthly_budget_usd=0.003)

        missing = [chat(client, secret, model="nope").status_code for _ in range(5)]
        ok = chat(client, secret).status_code

    assert missing == [404] * 5
    assert ok == 200


def test_streams_are_charged_from_the_reported_usage() -> None:
    provider = FakeProvider(stream_usage=True)  # 10 prompt + 5 completion = $0.002
    with make_client(provider) as client:
        secret = make_key(client, monthly_budget_usd=0.01)

        statuses = [chat(client, secret, stream=True).status_code for _ in range(7)]

    assert statuses == [200] * 5 + [429] * 2


def test_streams_without_usage_are_charged_from_the_text_sent() -> None:
    provider = FakeProvider(reply="one two three four")  # 18 chars: 5 tokens; prompt 8 estimated
    with make_client(provider) as client:
        secret = make_key(client, monthly_budget_usd=0.0021)  # one request at 8*100+5*200=1800µ$

        first = chat(client, secret, stream=True)
        second = chat(client, secret, stream=True)

    assert first.status_code == 200
    assert second.status_code == 429


def test_a_stream_that_cannot_start_costs_nothing() -> None:
    provider = FakeProvider(fail_first=99)
    with make_client(provider, retry_max_attempts=1) as client:
        secret = make_key(client, monthly_budget_usd=0.003)

        failures = [chat(client, secret, stream=True).status_code for _ in range(8)]

    assert failures == [502] * 8


def test_a_refused_budget_gives_back_the_reserved_tokens() -> None:
    with make_client() as client:
        secret = make_key(client, monthly_budget_usd=0.000001, tpm_limit=1_000)

        refused = [chat(client, secret).status_code for _ in range(10)]
        app: FastAPI = client.app  # type: ignore[assignment]
        assert client.portal is not None

        async def tokens_left() -> float:
            redis = app.state.redis
            keys = [k async for k in redis.scan_iter("*:tpm:*")]
            return float(await redis.hget(keys[0], "tokens"))

        left = client.portal.call(tokens_left)

    assert refused == [429] * 10
    assert left >= 995  # the token reservations were returned, not 10 x 520 consumed


def test_the_sdk_sees_a_rate_limit_error() -> None:
    with make_client() as client:
        secret = make_key(client, monthly_budget_usd=0.002)
        sdk = OpenAI(
            base_url="http://testserver/v1", api_key=secret, http_client=client, max_retries=0
        )
        sdk.chat.completions.create(
            model="test-model", messages=[{"role": "user", "content": "Hi"}], max_tokens=5
        )

        with pytest.raises(RateLimitError) as excinfo:
            sdk.chat.completions.create(
                model="test-model", messages=[{"role": "user", "content": "Hi"}], max_tokens=5
            )

    assert excinfo.value.code == "budget_exceeded"


def test_requests_still_work_when_redis_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    unreachable = Settings(_env_file=None, redis_url="redis://127.0.0.1:1/0", redis_timeout=0.2)
    monkeypatch.setattr(
        "app.lifespan.build_redis_client", lambda _: build_redis_client(unreachable)
    )

    with make_client() as client:
        secret = make_key(client, monthly_budget_usd=0.000001)

        statuses = [chat(client, secret).status_code for _ in range(3)]

    assert statuses == [200] * 3  # fails open


@pytest.mark.parametrize("backend", ["fake", "real"])
def test_concurrent_requests_through_the_whole_stack_cannot_overspend(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """12 slow requests race for a budget that covers 4 worst-case reservations."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    if backend == "real":
        url = os.environ.get("LLMROUTER_TEST_REDIS_URL")
        if not url:
            pytest.skip("set LLMROUTER_TEST_REDIS_URL to test against a real Redis")
        real = Settings(_env_file=None, redis_url=url)
        monkeypatch.setattr("app.lifespan.build_redis_client", lambda _: build_redis_client(real))

    slow = FakeProvider(latency=0.4)  # keeps every reservation outstanding while the rest arrive
    with make_client(slow) as client:
        app: FastAPI = client.app  # type: ignore[assignment]
        assert client.portal is not None
        client.portal.call(app.state.redis.flushdb)
        secret = make_key(
            client, monthly_budget_usd=0.0075
        )  # four worst cases: 4 * $0.0018 = $0.0072

        with ThreadPoolExecutor(max_workers=12) as pool:
            statuses = list(pool.map(lambda _: chat(client, secret).status_code, range(12)))

        client.portal.call(app.state.redis.flushdb)

    assert statuses.count(200) == 4
    assert statuses.count(429) == 8


# ---------- configuring budgets ----------


def test_the_admin_api_sets_and_shows_budgets() -> None:
    with make_client() as client:
        headers = admin(client)
        tenant = client.post(
            "/admin/tenants", json={"name": "globex", "monthly_budget_usd": 250.5}, headers=headers
        ).json()
        key = client.post(
            f"/admin/tenants/{tenant['id']}/keys",
            json={"name": "app", "monthly_budget_usd": 12.25},
            headers=headers,
        ).json()
        plain = client.post(
            f"/admin/tenants/{tenant['id']}/keys", json={"name": "plain"}, headers=headers
        ).json()

    assert tenant["monthly_budget_usd"] == 250.5
    assert key["monthly_budget_usd"] == 12.25
    assert plain["monthly_budget_usd"] is None


def test_a_tenant_budget_can_be_changed_and_removed() -> None:
    with make_client() as client:
        headers = admin(client)
        tenant = client.post("/admin/tenants", json={"name": "globex"}, headers=headers).json()[
            "id"
        ]
        url = f"/admin/tenants/{tenant}"

        set_ = client.patch(url, json={"monthly_budget_usd": 100}, headers=headers).json()
        renamed = client.patch(url, json={"name": "initech"}, headers=headers).json()
        cleared = client.patch(url, json={"monthly_budget_usd": None}, headers=headers).json()

    assert set_["monthly_budget_usd"] == 100
    assert renamed["monthly_budget_usd"] == 100  # leaving it out keeps it
    assert cleared["monthly_budget_usd"] is None


def test_raising_a_budget_takes_effect_immediately() -> None:
    with make_client() as client:
        headers = admin(client)
        tenant = client.get("/admin/tenants", headers=headers).json()["data"][0]["id"]
        client.patch(
            f"/admin/tenants/{tenant}", json={"monthly_budget_usd": 0.002}, headers=headers
        )
        assert chat(client, client.secret).status_code == 200
        assert chat(client, client.secret).status_code == 429

        client.patch(f"/admin/tenants/{tenant}", json={"monthly_budget_usd": 0.01}, headers=headers)

        assert chat(client, client.secret).status_code == 200


@pytest.mark.parametrize("value", [0, -1, 1_000_001, "lots"])
def test_invalid_budgets_are_rejected(value: Any) -> None:
    with make_client() as client:
        headers = admin(client)
        tenant = client.post("/admin/tenants", json={"name": "globex"}, headers=headers).json()[
            "id"
        ]

        key = client.post(
            f"/admin/tenants/{tenant}/keys",
            json={"name": "k", "monthly_budget_usd": value},
            headers=headers,
        )
        new_tenant = client.post(
            "/admin/tenants", json={"name": "x", "monthly_budget_usd": value}, headers=headers
        )

    assert key.status_code == new_tenant.status_code == 400


def test_rotation_keeps_the_budget_unless_told_otherwise() -> None:
    with make_client() as client:
        headers = admin(client)
        tenant = client.post("/admin/tenants", json={"name": "globex"}, headers=headers).json()[
            "id"
        ]
        base = f"/admin/tenants/{tenant}/keys"
        keys = [
            client.post(
                base, json={"name": f"k{n}", "monthly_budget_usd": 5}, headers=headers
            ).json()
            for n in range(3)
        ]

        kept = client.post(f"{base}/{keys[0]['id']}/rotate", json={}, headers=headers).json()
        changed = client.post(
            f"{base}/{keys[1]['id']}/rotate", json={"monthly_budget_usd": 9}, headers=headers
        ).json()
        removed = client.post(
            f"{base}/{keys[2]['id']}/rotate", json={"monthly_budget_usd": None}, headers=headers
        ).json()

    assert (kept["monthly_budget_usd"], changed["monthly_budget_usd"]) == (5, 9)
    assert removed["monthly_budget_usd"] is None


def test_an_empty_tenant_update_is_still_rejected() -> None:
    with make_client() as client:
        headers = admin(client)
        tenant = client.post("/admin/tenants", json={"name": "globex"}, headers=headers).json()[
            "id"
        ]

        response = client.patch(f"/admin/tenants/{tenant}", json={}, headers=headers)

    assert response.status_code == 400


def test_migration_leaves_existing_rows_uncapped(tmp_path: Path) -> None:
    database = tmp_path / "budget.db"
    config = alembic_config(database)
    command.upgrade(config, "0006")
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
        (tenant,) = connection.execute("SELECT monthly_budget_nano_usd FROM tenants").fetchone()
        (key,) = connection.execute("SELECT monthly_budget_nano_usd FROM api_keys").fetchone()
    assert tenant is None
    assert key is None


def test_the_budget_migration_can_be_reversed(tmp_path: Path) -> None:
    database = tmp_path / "budget.db"
    config = alembic_config(database)
    command.upgrade(config, "head")

    command.downgrade(config, "0006")

    with sqlite3.connect(database) as connection:
        tenant_columns = [row[1] for row in connection.execute("PRAGMA table_info(tenants)")]
        key_columns = [row[1] for row in connection.execute("PRAGMA table_info(api_keys)")]
    assert "monthly_budget_nano_usd" not in tenant_columns + key_columns
    assert "tpm_limit" in key_columns
