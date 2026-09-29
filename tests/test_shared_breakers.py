import asyncio
import logging
import random
from collections.abc import Sequence
from typing import Any

import fakeredis
import httpx
import pytest
from redis.asyncio import Redis

from app.core.config import Settings
from app.main import create_app
from app.providers.errors import ProviderDown
from app.providers.fake import FakeProvider
from app.resilience.board import BreakerBoard, BreakerSettings, DeploymentUnavailable
from app.resilience.breaker import CircuitBreaker, CircuitState
from app.resilience.shared import SharedBreakerStore
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry
from app.router.router import Router
from app.schemas.chat import ChatCompletionRequest, Message
from tests.conftest import TEST_ALIAS, AuthedClient, model_config, registry_with

CLOSED, OPEN, HALF_OPEN = CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN
A_URL, B_URL = "https://a.example.com/v1", "https://b.example.com/v1"
REQUEST = ChatCompletionRequest(model=TEST_ALIAS, messages=[Message(role="user", content="hi")])
DOWN = ProviderDown("down")


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FirstChoice(random.Random):
    """Always picks the first candidate, so a test knows which deployment is tried first."""

    def choices(  # type: ignore[override]
        self, population: Sequence[Any], weights: Any = None, **kwargs: Any
    ) -> list[Any]:
        return [population[0]]


# ---------- the store ----------


async def test_an_open_breaker_is_visible_with_its_remaining_cooldown(redis_client: Redis) -> None:
    store = SharedBreakerStore(redis_client, "t:")
    assert await store.remaining_open("x") is None

    assert await store.publish_open("x", 30)

    remaining = await store.remaining_open("x")
    assert remaining is not None
    assert 29 < remaining <= 30


async def test_it_disappears_when_the_cooldown_ends(redis_client: Redis) -> None:
    store = SharedBreakerStore(redis_client, "t:")
    await store.publish_open("x", 0.05)
    await asyncio.sleep(0.15)
    assert await store.remaining_open("x") is None


async def test_closing_removes_it(redis_client: Redis) -> None:
    store = SharedBreakerStore(redis_client, "t:")
    await store.publish_open("x", 30)
    assert await store.publish_closed("x")
    assert await store.remaining_open("x") is None


async def test_names_do_not_collide(redis_client: Redis) -> None:
    store = SharedBreakerStore(redis_client, "t:")
    await store.publish_open("one", 30)
    assert await store.remaining_open("two") is None
    other = SharedBreakerStore(redis_client, "other:")
    assert await other.remaining_open("one") is None


async def test_a_zero_cooldown_publishes_nothing(redis_client: Redis) -> None:
    store = SharedBreakerStore(redis_client, "t:")
    assert not await store.publish_open("x", 0)
    assert await redis_client.keys("t:*") == []


async def test_breaker_names_with_awkward_characters_work(redis_client: Redis) -> None:
    store = SharedBreakerStore(redis_client, "t:")
    name = "provider:openai_compatible@https://api.example.com/v1"
    await store.publish_open(name, 30)
    assert await store.remaining_open(name) is not None


class BrokenRedis:
    """Every command fails, and the calls are counted."""

    def __init__(self) -> None:
        self.calls = 0

    async def set(self, *args: Any, **kwargs: Any) -> None:
        self.calls += 1
        raise ConnectionError("down")

    delete = pttl = set


async def test_a_broken_redis_means_nothing_is_known_and_nothing_is_raised() -> None:
    redis = BrokenRedis()
    store = SharedBreakerStore(redis, "t:")  # type: ignore[arg-type]

    assert not await store.publish_open("x", 30)
    assert await store.remaining_open("x") is None
    assert not await store.publish_closed("x")


async def test_it_stays_away_from_a_broken_redis_until_the_retry_time() -> None:
    clock = Clock()
    redis = BrokenRedis()
    store = SharedBreakerStore(redis, "t:", retry_after=5, clock=clock)  # type: ignore[arg-type]

    await store.remaining_open("x")
    await store.remaining_open("x")
    await store.publish_open("x", 30)
    assert redis.calls == 1
    assert not store.available

    clock.advance(5)
    assert store.available
    await store.remaining_open("x")
    assert redis.calls == 2


async def test_one_warning_per_spell_not_one_per_call(caplog: pytest.LogCaptureFixture) -> None:
    clock = Clock()
    store = SharedBreakerStore(BrokenRedis(), "t:", retry_after=5, clock=clock)  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING, logger="app.resilience.shared"):
        for _ in range(5):
            await store.remaining_open("x")
        clock.advance(5)
        await store.remaining_open("x")
    assert len([r for r in caplog.records if "unavailable" in r.message]) == 2


# ---------- breaker.trip ----------


def test_trip_opens_for_only_the_remaining_cooldown() -> None:
    clock = Clock()
    breaker = CircuitBreaker("x", cooldown=30, clock=clock)

    assert breaker.trip(12, "elsewhere")

    assert breaker.state is OPEN
    assert breaker.remaining_cooldown() == pytest.approx(12)
    assert breaker.last_trip_reason == "elsewhere"
    clock.advance(12)
    assert breaker.state is HALF_OPEN


def test_trip_does_nothing_to_a_breaker_that_is_already_open() -> None:
    clock = Clock()
    breaker = CircuitBreaker("x", failure_threshold=1, cooldown=30, clock=clock)
    breaker.acquire().failure()
    clock.advance(10)

    assert not breaker.trip(30, "elsewhere")
    assert breaker.remaining_cooldown() == pytest.approx(20)  # not extended
    assert breaker.last_trip_reason == "1 consecutive failures"


def test_trip_can_reopen_a_half_open_breaker() -> None:
    clock = Clock()
    breaker = CircuitBreaker("x", failure_threshold=1, cooldown=10, clock=clock)
    breaker.acquire().failure()
    clock.advance(10)
    assert breaker.state is HALF_OPEN

    assert breaker.trip(4, "elsewhere")
    assert breaker.state is OPEN


def test_a_tripped_breaker_ignores_calls_started_before() -> None:
    breaker = CircuitBreaker("x", failure_threshold=1)
    straggler = breaker.acquire()
    breaker.trip(30, "elsewhere")
    straggler.success()
    assert breaker.state is OPEN


# ---------- replicas sharing state ----------


def deployment(
    provider: FakeProvider | None = None, *, model: str = "m", url: str = A_URL
) -> Deployment:
    return Deployment.create(model_config(model=model, base_url=url), provider or FakeProvider())


class Replica:
    """One gateway process: its own board and clock, a shared Redis."""

    def __init__(self, redis: Redis, *, interval: float = 0.0, cooldown: float = 30.0) -> None:
        self.clock = Clock()
        self.store = SharedBreakerStore(redis, "t:", clock=self.clock)
        self.board = BreakerBoard(
            BreakerSettings(failure_threshold=2, failure_rate_threshold=None, cooldown=cooldown),
            clock=self.clock,
            shared=self.store,
            sync_interval=interval,
        )

    async def fail(self, target: Deployment, times: int = 2) -> None:
        for _ in range(times):
            self.board.acquire(target).failure(DOWN)
        await self.board.drain()


async def test_a_breaker_opened_on_one_replica_opens_on_the_others() -> None:
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    one, two = Replica(redis), Replica(redis)
    target = deployment()

    await one.fail(target)
    assert not two.board.is_open(target)  # it has not looked yet
    await two.board.sync(target)

    assert two.board.is_open(target)
    with pytest.raises(DeploymentUnavailable) as caught:
        two.board.acquire(target)
    assert 0 < caught.value.retry_after <= 30


async def test_the_provider_breaker_is_shared_so_other_models_are_protected() -> None:
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    one, two = Replica(redis), Replica(redis)

    await one.fail(deployment(model="first"))
    other_model = deployment(model="second")  # never failed anywhere
    await two.board.sync(other_model)

    assert two.board.provider_breaker(other_model).state is OPEN
    assert two.board.deployment_breaker(other_model).state is CLOSED


async def test_an_unrelated_vendor_is_unaffected() -> None:
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    one, two = Replica(redis), Replica(redis)
    await one.fail(deployment(url=A_URL))

    elsewhere = deployment(url=B_URL)
    await two.board.sync(elsewhere)

    assert not two.board.is_open(elsewhere)


async def test_the_borrowed_cooldown_is_what_remains_not_a_fresh_one() -> None:
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    one, two = Replica(redis, cooldown=30), Replica(redis, cooldown=30)
    target = deployment()
    await one.fail(target)
    await redis.pexpire("t:" + one.board.deployment_breaker(target).name, 8_000)

    await two.board.sync(target)

    assert two.board.deployment_breaker(target).remaining_cooldown() == pytest.approx(8, abs=0.5)


async def test_a_borrowed_opening_is_not_announced_again() -> None:
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    one, two = Replica(redis), Replica(redis)
    target = deployment()
    await one.fail(target)
    name = one.board.deployment_breaker(target).name
    await redis.pexpire("t:" + name, 5_000)

    await two.board.sync(target)
    await two.board.drain()

    assert await redis.pttl("t:" + name) <= 5_000  # replica two did not extend it


async def test_a_recovery_is_shared_too() -> None:
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    one, two = Replica(redis), Replica(redis)
    target = deployment()
    await one.fail(target)
    one.clock.advance(30)
    one.board.acquire(target).success()  # the trial succeeds: closed
    await one.board.drain()

    await two.board.sync(target)

    assert not two.board.is_open(target)


async def test_a_replica_looks_at_redis_only_once_per_interval() -> None:
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    two = Replica(redis, interval=1.0)
    looks = 0
    original = two.store.remaining_open

    async def counting(name: str) -> float | None:
        nonlocal looks
        looks += 1
        return await original(name)

    two.store.remaining_open = counting  # type: ignore[method-assign]
    target = deployment()

    for _ in range(5):
        await two.board.sync(target)
    assert looks == 2  # the provider's breaker and the deployment's, once each

    two.clock.advance(1)
    await two.board.sync(target)
    assert looks == 4


async def test_it_does_not_ask_about_a_breaker_that_is_already_open_locally() -> None:
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    two = Replica(redis)
    target = deployment()
    await two.fail(target)
    looks = 0
    original = two.store.remaining_open

    async def counting(name: str) -> float | None:
        nonlocal looks
        looks += 1
        return await original(name)

    two.store.remaining_open = counting  # type: ignore[method-assign]
    await two.board.sync(target)
    assert looks == 0


async def test_without_a_shared_store_sync_does_nothing() -> None:
    board = BreakerBoard(BreakerSettings(failure_threshold=1, failure_rate_threshold=None))
    target = deployment()
    await board.sync(target)
    board.acquire(target).failure(DOWN)
    await board.sync(target)
    await board.drain()
    assert board.is_open(target)


def test_breakers_driven_without_an_event_loop_still_work_with_a_shared_store() -> None:
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    replica = Replica(redis)
    target = deployment()
    for _ in range(2):
        replica.board.acquire(target).failure(DOWN)  # no running loop: nothing to announce on
    assert replica.board.is_open(target)


async def test_a_broken_redis_leaves_the_local_breakers_working() -> None:
    clock = Clock()
    board = BreakerBoard(
        BreakerSettings(failure_threshold=2, failure_rate_threshold=None),
        clock=clock,
        shared=SharedBreakerStore(BrokenRedis(), "t:", clock=clock),  # type: ignore[arg-type]
        sync_interval=0,
    )
    target = deployment()

    await board.sync(target)
    for _ in range(2):
        board.acquire(target).failure(DOWN)
    await board.drain()
    await board.sync(target)

    assert board.is_open(target)


async def test_the_transition_observer_still_hears_about_everything() -> None:
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    seen: list[str] = []
    board = BreakerBoard(
        BreakerSettings(failure_threshold=1, failure_rate_threshold=None),
        shared=SharedBreakerStore(redis, "t:"),
        on_transition=lambda name, old, new: seen.append(new.value),
    )
    board.acquire(deployment()).failure(DOWN)
    await board.drain()
    assert seen == ["open", "open"]


# ---------- through the router ----------


async def test_a_second_replica_skips_a_deployment_the_first_found_broken() -> None:
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    one, two = Replica(redis), Replica(redis)

    def router(replica: Replica, bad: FakeProvider, good: FakeProvider) -> Router:
        return Router(
            ModelRegistry(
                [
                    deployment(bad, model="bad", url=A_URL),
                    deployment(good, model="good", url=B_URL),
                ]
            ),
            breakers=replica.board,
            rng=FirstChoice(),
        )

    bad_one, good_one = FakeProvider(fail_first=999), FakeProvider(reply="ok")
    first = router(one, bad_one, good_one)
    await one.fail(first.registry.get(TEST_ALIAS)[0])

    bad_two, good_two = FakeProvider(fail_first=999), FakeProvider(reply="ok")
    second = router(two, bad_two, good_two)
    response = await second.complete(REQUEST)

    assert response.choices[0].message.content == "ok"
    assert bad_two.calls == 0  # replica two never had to find out the hard way


# ---------- through the app ----------


def api_client(provider: FakeProvider, **settings: Any) -> AuthedClient:
    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return registry_with(provider)

    options = {"retry_base_delay": 0.0, "retry_max_attempts": 1, **settings}
    return AuthedClient(create_app(Settings(_env_file=None, **options), factory))


def breaker_keys(client: AuthedClient) -> list[str]:
    async def look() -> list[str]:
        await client.app.state.router.breakers.drain()  # type: ignore[attr-defined]
        keys = await client.app.state.redis.keys("llmrouter:breaker:*")  # type: ignore[attr-defined]
        return sorted(keys)

    assert client.portal is not None
    return client.portal.call(look)  # type: ignore[no-any-return]


def ask(client: AuthedClient) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        json={"model": TEST_ALIAS, "messages": [{"role": "user", "content": "hi"}]},
    )


def test_the_app_announces_an_open_breaker_through_redis() -> None:
    with api_client(FakeProvider(fail_first=999), breaker_failure_threshold=2) as client:
        ask(client)
        ask(client)
        keys = breaker_keys(client)

    assert len(keys) == 2
    assert any(":provider:" in key for key in keys)
    assert any(":deployment:" in key for key in keys)


def test_sharing_can_be_turned_off() -> None:
    with api_client(
        FakeProvider(fail_first=999), breaker_failure_threshold=2, breaker_shared=False
    ) as client:
        ask(client)
        ask(client)
        assert breaker_keys(client) == []
