import asyncio
import logging
import statistics
import threading
import time
from collections.abc import Iterator
from typing import Any, cast

import fakeredis
import httpx
import pytest
from openai import InternalServerError, OpenAI
from pydantic import ValidationError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError

from app.core.config import Settings
from app.core.redis import build_redis_client
from app.providers.fake import FakeProvider
from app.quota.bucket import BucketStore, TakeResult
from app.quota.budget import BudgetStore, SpendResult
from app.quota.budget_limiter import BudgetLimiter
from app.quota.guard import FailureMode, QuotaGuard, QuotaUnavailableError
from app.quota.limiter import RateLimiter
from app.quota.quota import reserve_request_quota
from app.quota.reservation import NO_RESERVATION
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry
from tests.conftest import AuthedClient, model_config
from tests.test_budget_limit import PRICES
from tests.test_budget_limit import request as chat_request
from tests.test_rpm_limit import auth as limited_auth
from tests.test_rpm_limit import chat, make_client


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Backend:
    """A stand-in for a Redis operation that fails on demand and counts how often it is asked."""

    def __init__(self) -> None:
        self.calls = 0
        self.error: Exception | None = None

    async def __call__(self) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return "result"


def guard(
    mode: str = "open", threshold: int = 3, cooldown: float = 5.0, clock: Clock | None = None
) -> QuotaGuard:
    return QuotaGuard(
        cast(FailureMode, mode),
        failure_threshold=threshold,
        cooldown=cooldown,
        clock=clock or Clock(),
    )


def down() -> Backend:
    backend = Backend()
    backend.error = RedisConnectionError("down")
    return backend


# ---------- the guard ----------


async def test_a_healthy_backend_passes_straight_through() -> None:
    backend = Backend()

    assert await guard().enforce(backend, None, what="x") == "result"
    assert backend.calls == 1


async def test_failing_open_returns_the_fallback_and_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="app.quota.guard"):
        outcome = await guard("open").enforce(down(), "UNENFORCED", what="rate limiting")

    assert outcome == "UNENFORCED"
    assert any(
        "rate limiting unavailable, allowing request" in r.getMessage() for r in caplog.records
    )


async def test_failing_closed_refuses(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="app.quota.guard"):
        with pytest.raises(QuotaUnavailableError) as excinfo:
            await guard("closed").enforce(down(), None, what="token limiting")

    assert excinfo.value.retry_after >= 1
    assert "temporarily unavailable" in str(excinfo.value)
    assert any(
        "token limiting unavailable, refusing request" in r.getMessage() for r in caplog.records
    )


@pytest.mark.parametrize(
    "error",
    [
        RedisConnectionError("x"),
        RedisError("x"),
        OSError("x"),
        TimeoutError("x"),
        ConnectionResetError(),
    ],
)
async def test_every_kind_of_connection_trouble_counts(error: Exception) -> None:
    backend = Backend()
    backend.error = error

    assert await guard("open").enforce(backend, "fallback", what="x") == "fallback"


async def test_programming_errors_are_not_mistaken_for_an_outage() -> None:
    backend = Backend()
    backend.error = ValueError("a bug")

    for mode in ("open", "closed"):
        with pytest.raises(ValueError, match="a bug"):
            await guard(mode).enforce(backend, None, what="x")


# ---------- the circuit breaker ----------


@pytest.mark.parametrize("mode", ["open", "closed"])
async def test_after_repeated_failures_redis_is_left_alone(mode: str) -> None:
    backend = down()
    breaker = guard(mode, threshold=3)

    async def attempt() -> object:
        try:
            return await breaker.enforce(backend, "fallback", what="x")
        except QuotaUnavailableError:
            return "refused"

    outcomes = [await attempt() for _ in range(10)]

    assert backend.calls == 3  # then it stopped asking
    assert outcomes == (["fallback"] * 10 if mode == "open" else ["refused"] * 10)
    assert breaker.paused


async def test_the_pause_ends_with_one_probe_and_a_success_resumes_service() -> None:
    clock, backend = Clock(), down()
    breaker = guard("open", threshold=2, cooldown=5, clock=clock)
    for _ in range(2):
        await breaker.enforce(backend, None, what="x")
    assert breaker.paused

    clock.now += 5.1  # the pause is over
    backend.error = None
    probe = await breaker.enforce(backend, None, what="x")

    assert probe == "result"
    assert not breaker.paused
    assert backend.calls == 3
    await breaker.enforce(backend, None, what="x")  # and it is back to normal
    assert backend.calls == 4


async def test_a_failed_probe_pauses_again_immediately() -> None:
    clock, backend = Clock(), down()
    breaker = guard("open", threshold=3, cooldown=5, clock=clock)
    for _ in range(3):
        await breaker.enforce(backend, None, what="x")
    calls_when_tripped = backend.calls

    clock.now += 5.1
    await breaker.enforce(backend, None, what="x")  # one probe, which fails
    await breaker.enforce(backend, None, what="x")  # skipped again

    assert backend.calls == calls_when_tripped + 1
    assert breaker.paused


async def test_the_retry_hint_counts_down_with_the_pause() -> None:
    clock, backend = Clock(), down()
    breaker = guard("closed", threshold=1, cooldown=20, clock=clock)
    with pytest.raises(QuotaUnavailableError):
        await breaker.enforce(backend, None, what="x")  # trips it
    clock.now += 12

    with pytest.raises(QuotaUnavailableError) as excinfo:
        await breaker.enforce(backend, None, what="x")

    assert excinfo.value.retry_after == pytest.approx(8.0)


async def test_scattered_failures_never_trip_it() -> None:
    backend = Backend()
    breaker = guard("open", threshold=3)

    for _ in range(10):
        backend.error = RedisConnectionError("blip")
        await breaker.enforce(backend, None, what="x")
        await breaker.enforce(backend, None, what="x")
        backend.error = None
        await breaker.enforce(backend, None, what="x")  # a success resets the count

    assert not breaker.paused
    assert backend.calls == 30


async def test_a_zero_cooldown_never_pauses() -> None:
    backend = down()
    breaker = guard("open", threshold=1, cooldown=0)

    for _ in range(5):
        await breaker.enforce(backend, None, what="x")

    assert backend.calls == 5


async def test_a_storm_of_failures_trips_it_and_logs_stay_quiet(
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = down()
    breaker = guard("open", threshold=3)

    with caplog.at_level(logging.INFO, logger="app.quota.guard"):
        await asyncio.gather(*(breaker.enforce(backend, None, what="x") for _ in range(5)))
        for _ in range(200):
            await breaker.enforce(backend, None, what="x")  # all skipped

    assert breaker.paused
    assert backend.calls < 10
    messages = [r.getMessage() for r in caplog.records]
    assert sum("not calling it" in m for m in messages) == 1  # announced once, not 200 times
    assert sum("unavailable" in m for m in messages) <= 6  # only real attempts log


async def test_recovery_is_announced_once(caplog: pytest.LogCaptureFixture) -> None:
    clock, backend = Clock(), down()
    breaker = guard("open", threshold=2, cooldown=1, clock=clock)
    for _ in range(2):
        await breaker.enforce(backend, None, what="x")
    clock.now += 2
    backend.error = None

    with caplog.at_level(logging.INFO, logger="app.quota.guard"):
        for _ in range(5):
            await breaker.enforce(backend, None, what="x")

    assert sum("recovered" in r.getMessage() for r in caplog.records) == 1


# ---------- settling bookkeeping never fails a served request ----------


@pytest.mark.parametrize("mode", ["open", "closed"])
async def test_settling_never_raises_in_either_mode(
    mode: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="app.quota.guard"):
        outcome = await guard(mode).settle(down(), what="token reservation")

    assert outcome is None
    assert any("could not settle token reservation" in r.getMessage() for r in caplog.records)


async def test_settling_counts_towards_the_pause_and_skips_while_paused() -> None:
    backend = down()
    breaker = guard("open", threshold=2)

    for _ in range(5):
        await breaker.settle(backend, what="x")

    assert backend.calls == 2 and breaker.paused


async def test_settling_returns_the_result_when_healthy() -> None:
    assert await guard().settle(Backend(), what="x") == "result"


@pytest.mark.parametrize(
    "kwargs",
    [{"mode": "maybe"}, {"failure_threshold": 0}, {"cooldown": -1}],
)
def test_the_guard_rejects_nonsense_settings(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        QuotaGuard(**kwargs)


# ---------- the limiters under each policy ----------


class BrokenBuckets(BucketStore):
    def __init__(self) -> None:
        self.calls = 0

    async def take(self, *args: Any, **kwargs: Any) -> TakeResult:
        self.calls += 1
        raise RedisConnectionError("down")

    async def adjust(self, *args: Any, **kwargs: Any) -> int:
        raise RedisConnectionError("down")


class BrokenBudgets(BudgetStore):
    def __init__(self) -> None:
        self.calls = 0

    async def spend(self, *args: Any, **kwargs: Any) -> SpendResult:
        self.calls += 1
        raise RedisConnectionError("down")

    async def adjust(self, *args: Any, **kwargs: Any) -> dict[str, int]:
        raise RedisConnectionError("down")


def prices_registry() -> ModelRegistry:
    return ModelRegistry([Deployment.create(model_config(**PRICES), FakeProvider())])


async def test_failing_open_lets_every_kind_of_limited_request_through() -> None:
    open_guard = guard("open")
    limiter = RateLimiter(BrokenBuckets(), guard=open_guard)
    budgets = BudgetLimiter(BrokenBudgets(), guard=open_guard)
    caller = limited_auth(rpm_limit=1)
    tokens = caller.__class__(**{**caller.__dict__, "tpm_limit": 100, "key_budget_nano": 1})

    assert await limiter.check_request(tokens) is None
    assert await limiter.reserve_tokens(tokens, chat_request()) is NO_RESERVATION
    assert await budgets.reserve(tokens, chat_request(), prices_registry(), 8) is NO_RESERVATION


async def test_failing_closed_refuses_every_kind_of_limited_request() -> None:
    closed_guard = guard("closed", threshold=100)
    limiter = RateLimiter(BrokenBuckets(), guard=closed_guard)
    budgets = BudgetLimiter(BrokenBudgets(), guard=closed_guard)
    caller = limited_auth(rpm_limit=1)
    caller = caller.__class__(**{**caller.__dict__, "tpm_limit": 100, "key_budget_nano": 10**9})

    with pytest.raises(QuotaUnavailableError):
        await limiter.check_request(caller)
    with pytest.raises(QuotaUnavailableError):
        await limiter.reserve_tokens(caller, chat_request())
    with pytest.raises(QuotaUnavailableError):
        await budgets.reserve(caller, chat_request(), prices_registry(), 8)


async def test_keys_without_limits_never_touch_redis_even_when_failing_closed() -> None:
    buckets, budgets_store = BrokenBuckets(), BrokenBudgets()
    closed_guard = guard("closed")
    limiter = RateLimiter(buckets, guard=closed_guard)
    budgets = BudgetLimiter(budgets_store, guard=closed_guard)
    free = limited_auth(rpm_limit=None)

    assert await limiter.check_request(free) is None
    assert await limiter.reserve_tokens(free, chat_request()) is NO_RESERVATION
    assert await budgets.reserve(free, chat_request(), prices_registry(), 8) is NO_RESERVATION
    assert buckets.calls == budgets_store.calls == 0


async def test_a_reservation_whose_settlement_fails_closed_does_not_raise() -> None:
    reservation_store = _TokenStoreThatFailsToSettle()
    limiter = RateLimiter(reservation_store, guard=guard("closed", threshold=100))
    caller = limited_auth(rpm_limit=None)
    caller = caller.__class__(**{**caller.__dict__, "tpm_limit": 1_000})

    reservation = await limiter.reserve_tokens(caller, chat_request(max_tokens=92))
    await reservation.settle(10)  # the response is already produced: this must not fail it

    assert reservation.reserved == 100


class _TokenStoreThatFailsToSettle(BucketStore):
    def __init__(self) -> None:
        pass

    async def take(self, *args: Any, **kwargs: Any) -> TakeResult:
        return TakeResult(True, 900, 0.0)

    async def adjust(self, *args: Any, **kwargs: Any) -> int:
        raise RedisConnectionError("down")


class _RecordingTokens(BucketStore):
    def __init__(self) -> None:
        self.deltas: list[int] = []

    async def take(self, *args: Any, **kwargs: Any) -> TakeResult:
        return TakeResult(True, 900, 0.0)

    async def adjust(self, name: str, *, delta: int, **kwargs: Any) -> int:
        self.deltas.append(delta)
        return 1_000


async def test_tokens_already_reserved_come_back_if_the_budget_cannot_be_checked() -> None:
    shared = guard("closed", threshold=100)
    tokens = _RecordingTokens()
    limiter = RateLimiter(tokens, guard=shared)
    budgets = BudgetLimiter(BrokenBudgets(), guard=shared)
    caller = limited_auth(rpm_limit=None)
    caller = caller.__class__(**{**caller.__dict__, "tpm_limit": 1_000, "key_budget_nano": 10**9})

    with pytest.raises(QuotaUnavailableError):
        await reserve_request_quota(
            limiter, budgets, caller, chat_request(max_tokens=92), prices_registry()
        )

    assert tokens.deltas == [100]  # the 100 reserved tokens were returned


# ---------- the settings ----------


def test_the_policy_defaults_to_failing_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLMROUTER_QUOTA_FAILURE_MODE", raising=False)

    settings = Settings(_env_file=None)

    assert (settings.quota_failure_mode, settings.quota_breaker_threshold) == ("open", 3)
    assert settings.quota_breaker_cooldown == 5.0


def test_the_policy_can_be_chosen_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMROUTER_QUOTA_FAILURE_MODE", "closed")

    assert Settings(_env_file=None).quota_failure_mode == "closed"


def test_an_unknown_policy_is_rejected_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMROUTER_QUOTA_FAILURE_MODE", "whatever")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


# ---------- over HTTP ----------

BODY = {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]}


def unreachable_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    dead = Settings(_env_file=None, redis_url="redis://127.0.0.1:1/0", redis_timeout=0.2)
    monkeypatch.setattr("app.lifespan.build_redis_client", lambda _: build_redis_client(dead))


def chat_status(client: AuthedClient, secret: str, **extra: Any) -> httpx.Response:
    return chat(client, secret, **extra)


def test_failing_open_serves_limited_keys_unenforced(monkeypatch: pytest.MonkeyPatch) -> None:
    unreachable_redis(monkeypatch)
    with make_client() as client:
        rpm = client.issue_key(rpm_limit=1)
        tpm = client.issue_key(tpm_limit=300)
        budget = client.issue_key()
        statuses = [chat(client, rpm).status_code for _ in range(4)] + [
            chat(client, tpm).status_code for _ in range(4)
        ]
        del budget

    assert statuses == [200] * 8


def test_failing_closed_refuses_limited_keys_with_a_503(monkeypatch: pytest.MonkeyPatch) -> None:
    unreachable_redis(monkeypatch)
    provider = FakeProvider()
    with make_client(provider, quota_failure_mode="closed") as client:
        rpm = client.issue_key(rpm_limit=100)
        tpm = client.issue_key(tpm_limit=100_000)

        by_rate = chat(client, rpm)
        by_tokens = chat(client, tpm)

    for response in (by_rate, by_tokens):
        assert response.status_code == 503
        error = response.json()["error"]
        assert error["type"] == "api_error"
        assert error["code"] == "quota_unavailable"
        assert int(response.headers["Retry-After"]) >= 1
        assert not [h for h in response.headers if h.startswith("x-ratelimit-")]
    assert provider.calls == 0  # nothing reached a provider


def test_failing_closed_still_serves_keys_with_no_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    unreachable_redis(monkeypatch)
    with make_client(quota_failure_mode="closed") as client:
        unlimited = chat(client, client.secret)  # the default key has no limits
        models = client.get("/v1/models", headers=client.as_key(client.issue_key(rpm_limit=5)))
        health = client.get("/health")

    assert unlimited.status_code == 200
    assert models.status_code == 200  # listing models involves no quota
    assert health.status_code == 200


def test_the_gateway_default_limits_apply_when_failing_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unreachable_redis(monkeypatch)
    with make_client(quota_failure_mode="closed", default_rpm_limit=100) as client:
        response = chat(client, client.secret)  # a gateway-wide limit now covers every key

    assert response.status_code == 503


def test_failing_closed_refuses_streams_with_a_plain_503(monkeypatch: pytest.MonkeyPatch) -> None:
    unreachable_redis(monkeypatch)
    with make_client(quota_failure_mode="closed") as client:
        response = chat(client, client.issue_key(rpm_limit=100), stream=True)

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")


def test_the_sdk_sees_a_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    unreachable_redis(monkeypatch)
    with make_client(quota_failure_mode="closed") as client:
        sdk = OpenAI(
            base_url="http://testserver/v1",
            api_key=client.issue_key(rpm_limit=100),
            http_client=client,
            max_retries=0,
        )

        with pytest.raises(InternalServerError) as excinfo:
            sdk.chat.completions.create(
                model="test-model", messages=[{"role": "user", "content": "hi"}]
            )

    assert excinfo.value.status_code == 503
    assert excinfo.value.code == "quota_unavailable"


def test_a_storm_during_an_outage_does_not_flood_the_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    unreachable_redis(monkeypatch)
    with make_client() as client, caplog.at_level(logging.WARNING, logger="app.quota.guard"):
        secret = client.issue_key(rpm_limit=100)

        for _ in range(60):
            chat(client, secret)

    warnings = [r for r in caplog.records if r.name == "app.quota.guard"]
    assert len(warnings) <= 8  # a few real failures and one "pausing" notice, not 60 tracebacks


# ---------- a Redis that hangs ----------


@pytest.fixture
def blackhole() -> Iterator[int]:
    """A TCP server that accepts connections and never answers, like a firewalled Redis."""
    started, port = threading.Event(), []

    def run() -> None:
        async def main() -> None:
            async def hold(reader: Any, writer: Any) -> None:
                await asyncio.sleep(3600)

            server = await asyncio.start_server(hold, "127.0.0.1", 0)
            port.append(server.sockets[0].getsockname()[1])
            started.set()
            await server.serve_forever()

        asyncio.run(main())

    threading.Thread(target=run, daemon=True).start()
    started.wait(5)
    yield port[0]


def timed(client: AuthedClient, secret: str, count: int) -> list[float]:
    durations = []
    for _ in range(count):
        started = time.perf_counter()
        chat(client, secret)
        durations.append(time.perf_counter() - started)
    return durations


@pytest.mark.parametrize("mode", ["open", "closed"])
def test_a_hung_redis_slows_only_the_first_few_requests(
    mode: str, blackhole: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    hung = Settings(_env_file=None, redis_url=f"redis://127.0.0.1:{blackhole}/0", redis_timeout=0.3)
    monkeypatch.setattr("app.lifespan.build_redis_client", lambda _: build_redis_client(hung))

    with make_client(quota_failure_mode=mode, quota_breaker_cooldown=60) as client:
        secret = client.issue_key(rpm_limit=100)

        durations = timed(client, secret, 10)

    assert all(d >= 0.25 for d in durations[:3])  # each of these waited out the timeout
    # Then the gateway stopped waiting. Medians, not maxima: one request can be delayed by a busy
    # machine without that meaning the gateway waited for Redis.
    assert statistics.median(durations[3:]) < 0.15
    assert sum(durations[3:]) < 7 * 0.3 * 0.75  # well under what seven full timeouts would cost


# ---------- an outage and a recovery ----------


class FlakyRedis(fakeredis.FakeAsyncRedis):
    """A Redis that can be switched off and on."""

    down = False

    def _check(self) -> None:
        if self.down:
            raise RedisConnectionError("switched off")

    async def evalsha(self, *args: Any, **kwargs: Any) -> Any:
        self._check()
        return await super().evalsha(*args, **kwargs)

    async def script_load(self, *args: Any, **kwargs: Any) -> Any:
        self._check()
        return await super().script_load(*args, **kwargs)

    async def ping(self, *args: Any, **kwargs: Any) -> Any:
        self._check()
        return await super().ping(*args, **kwargs)


def flaky(monkeypatch: pytest.MonkeyPatch) -> FlakyRedis:
    redis = FlakyRedis(decode_responses=True)
    monkeypatch.setattr("app.lifespan.build_redis_client", lambda _: redis)
    return redis


FAST_BREAKER: dict[str, Any] = {"quota_breaker_threshold": 2, "quota_breaker_cooldown": 0.2}


def test_failing_open_enforces_again_once_redis_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = flaky(monkeypatch)
    with make_client(**FAST_BREAKER) as client:
        secret = client.issue_key(rpm_limit=2)

        redis.down = True
        during = [chat(client, secret).status_code for _ in range(6)]
        redis.down = False
        time.sleep(0.3)  # the breaker's pause ends
        after = [chat(client, secret).status_code for _ in range(4)]

    assert during == [200] * 6  # unenforced while it was down
    assert after == [200, 200, 429, 429]  # and enforced again, from a fresh allowance


def test_failing_closed_serves_again_once_redis_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = flaky(monkeypatch)
    with make_client(quota_failure_mode="closed", **FAST_BREAKER) as client:
        secret = client.issue_key(rpm_limit=100)

        redis.down = True
        during = [chat(client, secret).status_code for _ in range(4)]
        redis.down = False
        time.sleep(0.3)
        after = chat(client, secret).status_code

    assert during == [503] * 4
    assert after == 200


@pytest.mark.parametrize("mode", ["open", "closed"])
def test_an_outage_in_the_middle_of_a_request_does_not_fail_it(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    redis = flaky(monkeypatch)

    class DiesMidRequest(FakeProvider):
        async def chat(self, request: Any) -> Any:
            redis.down = True  # Redis goes away after the quota was reserved ...
            return await super().chat(request)

    with make_client(DiesMidRequest(), quota_failure_mode=mode, **FAST_BREAKER) as client:
        secret = client.issue_key(tpm_limit=1_000)

        response = chat(client, secret, max_tokens=92)  # ... so it cannot be settled
        redis.down = False

        async def tokens() -> float:
            keys = [k async for k in redis.scan_iter("*:tpm:*")]
            return float(str(await redis.hget(keys[0], "tokens")))

        assert client.portal is not None
        left = client.portal.call(tokens)

    assert response.status_code == 200  # the reply was already produced and is delivered
    assert 895 <= left <= 905  # the key stays charged the full 100-token estimate, never less


def test_requests_that_need_no_redis_are_unaffected_by_the_breaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = flaky(monkeypatch)
    with make_client(quota_failure_mode="closed", **FAST_BREAKER) as client:
        limited = client.issue_key(rpm_limit=100)
        redis.down = True
        for _ in range(5):
            assert chat(client, limited).status_code == 503  # trips the breaker

        free = [chat(client, client.secret).status_code for _ in range(5)]

    assert free == [200] * 5
