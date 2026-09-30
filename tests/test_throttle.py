import random
from collections.abc import Sequence
from typing import Any

import fakeredis
import httpx
import pytest

from app.core.config import Settings
from app.core.tiers import Tier
from app.main import create_app
from app.providers.errors import ProviderDown, RateLimited
from app.providers.fake import FakeProvider
from app.resilience.board import (
    BreakerBoard,
    BreakerSettings,
    DeploymentThrottled,
    DeploymentUnavailable,
)
from app.resilience.shared import SharedBreakerStore
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry
from app.router.retry import RetryPolicy
from app.router.router import Router
from app.schemas.chat import ChatCompletionRequest
from tests.conftest import AuthedClient, model_config

URLS = {n: f"https://{n}.example.com/v1" for n in "abcd"}


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Sleeps:
    def __init__(self, clock: Clock | None = None) -> None:
        self.calls: list[float] = []
        self._clock = clock

    async def sleep(self, seconds: float) -> None:
        self.calls.append(seconds)
        if self._clock is not None:
            self._clock.advance(seconds)


class FirstChoice(random.Random):
    def choices(  # type: ignore[override]
        self, population: Sequence[Any], weights: Any = None, **kwargs: Any
    ) -> list[Any]:
        return [population[0]]

    def uniform(self, a: float, b: float) -> float:
        return b


def dep(
    alias: str = "m",
    vendor: str = "a",
    *,
    provider: FakeProvider | None = None,
    model: str | None = None,
    tier: Tier | None = None,
    price: float = 0.0,
) -> Deployment:
    return Deployment.create(
        model_config(
            name=alias,
            model=model or f"{alias}-{vendor}",
            base_url=URLS[vendor],
            tier=tier,
            input_cost_per_mtok=price,
            output_cost_per_mtok=price,
        ),
        provider or FakeProvider(),
    )


def board(**settings: Any) -> tuple[BreakerBoard, Clock]:
    clock = Clock()
    options = {
        "failure_threshold": 3,
        "failure_rate_threshold": None,
        "cooldown": 30.0,
        "throttle_cooldown": 5.0,
        "throttle_max_cooldown": 60.0,
        **settings,
    }
    return BreakerBoard(BreakerSettings(**options), clock=clock), clock


def limited(b: BreakerBoard, target: Deployment, retry_after: float | None = None) -> None:
    b.acquire(target).failure(RateLimited("slow down", retry_after=retry_after))


def request(model: str = "m") -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {"model": model, "messages": [{"role": "user", "content": "hi"}]}
    )


# ---------- how long a deployment rests ----------


def test_the_vendors_retry_after_is_honoured() -> None:
    b, clock = board()
    target = dep()
    limited(b, target, retry_after=12)

    assert b.throttle_remaining(target) == pytest.approx(12)
    with pytest.raises(DeploymentThrottled) as caught:
        b.acquire(target)
    assert caught.value.retry_after == pytest.approx(12)
    clock.advance(11.9)
    assert b.is_open(target)
    clock.advance(0.2)
    assert not b.is_open(target)
    b.acquire(target).success()


def test_a_refusal_is_a_rate_limit_error_so_clients_see_a_429() -> None:
    b, _ = board()
    target = dep()
    limited(b, target, retry_after=4)
    with pytest.raises(RateLimited):
        b.acquire(target)


def test_a_very_long_retry_after_is_capped() -> None:
    b, _ = board(throttle_max_cooldown=60)
    target = dep()
    limited(b, target, retry_after=86_400)
    assert b.throttle_remaining(target) == pytest.approx(60)


@pytest.mark.parametrize("retry_after", [None, 0, -3])
def test_without_a_usable_retry_after_the_default_applies(retry_after: float | None) -> None:
    b, _ = board(throttle_cooldown=5)
    target = dep()
    limited(b, target, retry_after=retry_after)
    assert b.throttle_remaining(target) == pytest.approx(5)


def test_repeated_429s_without_a_retry_after_rest_for_longer_each_time() -> None:
    b, clock = board(throttle_cooldown=5, throttle_max_cooldown=60)
    target = dep()
    rests = []
    for _ in range(6):
        limited(b, target)
        rests.append(round(b.throttle_remaining(target)))
        clock.advance(100)  # the rest is over before the next call
    assert rests == [5, 10, 20, 40, 60, 60]


def test_a_success_ends_the_streak() -> None:
    b, clock = board(throttle_cooldown=5)
    target = dep()
    limited(b, target)
    clock.advance(5)
    limited(b, target)
    clock.advance(10)
    b.acquire(target).success()
    limited(b, target)
    assert b.throttle_remaining(target) == pytest.approx(5)


def test_a_shorter_429_does_not_cut_a_longer_rest_short() -> None:
    b, _ = board()
    target = dep()
    limited(b, target, retry_after=40)
    limited_target = b  # a second, overlapping 429 from a call already in flight
    limited_target.throttle(target, 3)
    assert b.throttle_remaining(target) == pytest.approx(40)


def test_expire_throttle_forgets_the_rest() -> None:
    b, _ = board()
    target = dep()
    limited(b, target, retry_after=40)
    b.expire_throttle(target)
    assert not b.is_open(target)


# ---------- what a 429 is not ----------


def test_a_429_never_counts_against_a_circuit_breaker() -> None:
    b, clock = board(failure_threshold=1)
    target = dep()
    for _ in range(10):
        limited(b, target, retry_after=1)
        clock.advance(1)
    assert b.deployment_breaker(target).state.value == "closed"
    assert b.provider_breaker(target).state.value == "closed"
    assert b.deployment_breaker(target).consecutive_failures == 0


def test_a_429_on_a_trial_call_frees_the_trial_slot() -> None:
    b, clock = board(failure_threshold=1, cooldown=10)
    target = dep()
    b.acquire(target).failure(ProviderDown("x"))
    clock.advance(10)
    limited(b, target, retry_after=2)
    assert b.deployment_breaker(target).probes_in_flight == 0
    assert b.deployment_breaker(target).state.value == "half_open"  # neither closed nor reopened


def test_only_the_throttled_deployment_rests() -> None:
    b, _ = board()
    one, two = dep(model="one"), dep(model="two")  # the same vendor
    limited(b, one, retry_after=20)
    assert b.is_open(one)
    assert not b.is_open(two)
    b.acquire(two).success()


def test_it_is_told_apart_from_a_breaker_being_open() -> None:
    b, _ = board(failure_threshold=1)
    resting, broken, both = dep(model="r"), dep(model="b", vendor="b"), dep(model="x", vendor="c")
    limited(b, resting, retry_after=20)
    b.acquire(broken).failure(ProviderDown("x"))
    b.acquire(both).failure(ProviderDown("x"))
    b.throttle(both, 20)

    assert b.is_throttled_only(resting)
    assert not b.is_throttled_only(broken)
    assert not b.is_throttled_only(both)


def test_the_wait_includes_the_rest() -> None:
    b, _ = board()
    target = dep()
    limited(b, target, retry_after=17)
    assert b.wait_for(target) == pytest.approx(17)


def test_a_permit_reports_only_once() -> None:
    b, _ = board()
    target = dep()
    permit = b.acquire(target)
    permit.failure(RateLimited("x", retry_after=9))
    permit.failure(RateLimited("x", retry_after=90))
    permit.success()
    assert b.throttle_remaining(target) == pytest.approx(9)


# ---------- other replicas ----------


class Replica:
    def __init__(self, redis: Any) -> None:
        self.clock = Clock()
        self.board = BreakerBoard(
            BreakerSettings(failure_threshold=1, failure_rate_threshold=None),
            clock=self.clock,
            shared=SharedBreakerStore(redis, "t:", clock=self.clock),
            sync_interval=0.0,
        )


async def test_a_429_on_one_replica_rests_the_deployment_on_the_others() -> None:
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    one, two = Replica(redis), Replica(redis)
    target = dep()
    limited(one.board, target, retry_after=20)
    await one.board.drain()

    await two.board.sync(target)

    assert two.board.throttle_remaining(target) == pytest.approx(20, abs=1)
    assert two.board.deployment_breaker(target).state.value == "closed"  # it is not "broken"


async def test_a_borrowed_rest_is_not_announced_again() -> None:
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    one, two = Replica(redis), Replica(redis)
    target = dep()
    limited(one.board, target, retry_after=20)
    await one.board.drain()
    key = "t:throttle:" + target.id
    await redis.pexpire(key, 5_000)

    await two.board.sync(target)
    await two.board.drain()

    assert await redis.pttl(key) <= 5_000


async def test_a_replica_that_is_not_throttled_does_not_see_other_deployments_rest() -> None:
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    one, two = Replica(redis), Replica(redis)
    limited(one.board, dep(model="one"), retry_after=20)
    await one.board.drain()

    other = dep(model="other")
    await two.board.sync(other)

    assert not two.board.is_open(other)


# ---------- in the router ----------


def router_for(
    deployments: Sequence[Deployment],
    b: BreakerBoard,
    clock: Clock | None = None,
    fallbacks: dict[str, Sequence[str]] | None = None,
    attempts: int = 3,
) -> tuple[Router, Sleeps]:
    sleeps = Sleeps(clock)
    return (
        Router(
            ModelRegistry(deployments, fallbacks),
            policy=RetryPolicy(max_attempts=attempts, base_delay=0.5, max_delay=8.0),
            rng=FirstChoice(),
            sleep=sleeps.sleep,
            breakers=b,
        ),
        sleeps,
    )


async def test_a_throttled_deployment_is_left_alone_until_its_rest_is_over() -> None:
    b, clock = board()
    limiter = FakeProvider(fail_first=1, error_factory=lambda: RateLimited("x", retry_after=10))
    other = FakeProvider(reply="other")
    r, _ = router_for([dep(provider=limiter, vendor="a"), dep(provider=other, vendor="b")], b)

    for _ in range(5):
        await r.complete(request())  # the first 429 moves to the other; the rest skip the first
    assert limiter.calls == 1
    assert other.calls == 5

    clock.advance(9)
    await r.complete(request())
    assert limiter.calls == 1  # still resting

    clock.advance(1)
    await r.complete(request())
    assert limiter.calls == 2  # the rest is over, and it is tried again


async def test_a_429_uses_no_attempt_on_the_deployments_that_are_skipped_afterwards() -> None:
    b, _ = board()
    first, second, third = FakeProvider(), FakeProvider(fail_first=1), FakeProvider(reply="ok")
    r, _ = router_for(
        [
            dep(provider=first, vendor="a"),
            dep(provider=second, vendor="b"),
            dep(provider=third, vendor="c"),
        ],
        b,
        attempts=2,
    )
    limited(b, r.registry.get("m")[0], retry_after=30)

    response = await r.complete(request())

    assert response.choices[0].message.content == "ok"
    assert first.calls == 0


async def test_a_short_retry_after_on_the_only_deployment_is_waited_out_and_retried() -> None:
    b, clock = board()
    provider = FakeProvider(fail_first=1, error_factory=lambda: RateLimited("x", retry_after=3))
    target = dep(provider=provider)
    r, sleeps = router_for([target], b, clock)

    await r.complete(request())

    assert provider.calls == 2
    assert sleeps.calls == [3.0]
    assert not b.is_open(target)  # it was waited out, so it is not left resting


async def test_a_long_retry_after_on_the_only_deployment_is_passed_on_and_respected() -> None:
    b, clock = board()
    provider = FakeProvider(fail_first=1, error_factory=lambda: RateLimited("x", retry_after=30))
    r, sleeps = router_for([dep(provider=provider)], b, clock)

    with pytest.raises(RateLimited) as first:
        await r.complete(request())
    clock.advance(10)
    with pytest.raises(RateLimited) as later:
        await r.complete(request())

    assert first.value.retry_after == 30
    assert later.value.retry_after == pytest.approx(20)  # what is left of the vendor's 30
    assert provider.calls == 1  # the second request never reached the vendor
    assert sleeps.calls == []


async def test_when_every_model_is_resting_the_answer_is_a_429_with_the_soonest_time() -> None:
    b, clock = board()
    one, two = dep("m", "a"), dep("n", "b")
    r, _ = router_for([one, two], b, fallbacks={"m": ["n"]})
    limited(b, one, retry_after=30)
    clock.advance(10)
    limited(b, two, retry_after=30)

    with pytest.raises(DeploymentThrottled) as caught:
        await r.complete(request("m"))

    assert caught.value.retry_after == pytest.approx(20)  # the first is back soonest


async def test_a_mix_of_resting_and_broken_is_reported_as_an_outage() -> None:
    b, _ = board(failure_threshold=1)
    resting, broken = dep("m", "a"), dep("n", "b")
    r, _ = router_for([resting, broken], b, fallbacks={"m": ["n"]})
    limited(b, resting, retry_after=30)
    b.acquire(broken).failure(ProviderDown("x"))

    with pytest.raises(DeploymentUnavailable):
        await r.complete(request("m"))


async def test_streams_that_are_rate_limited_before_the_first_chunk_make_it_rest() -> None:
    b, _ = board()
    provider = FakeProvider(fail_first=1, error_factory=lambda: RateLimited("x", retry_after=30))
    target = dep(provider=provider)
    r, _ = router_for([target], b)

    with pytest.raises(RateLimited):
        [chunk async for chunk in r.stream(request())]

    assert b.throttle_remaining(target) == pytest.approx(30)


async def test_success_after_a_wait_clears_the_streak() -> None:
    b, clock = board(throttle_cooldown=5)
    provider = FakeProvider(fail_first=1, error_factory=lambda: RateLimited("x"))
    target = dep(provider=provider)
    r, _ = router_for([target], b, clock)

    await r.complete(request())  # a 429 with no Retry-After, waited out, then a success
    provider.fail_first = provider.calls + 1
    await r.complete(request())

    assert b.throttle_remaining(target) == 0  # the second 429 was waited out the same way


# ---------- smart routing and the API ----------


def api_client(deployments: list[Deployment], **settings: Any) -> AuthedClient:
    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return ModelRegistry(deployments)

    options = {"retry_base_delay": 0.0, "retry_max_attempts": 1, **settings}
    return AuthedClient(create_app(Settings(_env_file=None, **options), factory))


def post(client: AuthedClient, model: str) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
    )


def test_auto_stops_choosing_a_model_that_is_resting() -> None:
    limiter = FakeProvider(fail_first=1, error_factory=lambda: RateLimited("x", retry_after=30))
    cheap = dep("cheap", "a", provider=limiter, tier=Tier.SLM, price=0.05)
    middle = dep("middle", "b", tier=Tier.MID, price=0.6)
    with api_client([cheap, middle]) as client:
        responses = [post(client, "auto") for _ in range(4)]

    assert limiter.calls == 1
    later = responses[1:]
    assert {r.headers["x-llmrouter-model"] for r in later} == {"middle"}
    assert {r.headers["x-llmrouter-escalated"] for r in later} == {"false"}


def test_the_client_is_told_to_come_back_when_the_vendor_said_to() -> None:
    limiter = FakeProvider(fail_first=99, error_factory=lambda: RateLimited("x", retry_after=17))
    with api_client([dep(provider=limiter)]) as client:
        first = post(client, "m")
        second = post(client, "m")

    for response in (first, second):
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "upstream_rate_limited"
    assert first.headers["Retry-After"] == "17"
    assert 1 <= int(second.headers["Retry-After"]) <= 17
    assert limiter.calls == 1


@pytest.mark.parametrize("field", ["throttle_cooldown", "throttle_max_cooldown"])
def test_nonsense_throttle_settings_are_rejected(field: str) -> None:
    values: dict[str, Any] = {field: 0}
    with pytest.raises(ValueError):
        Settings(_env_file=None, **values)


async def test_a_deployment_that_starts_resting_mid_request_is_passed_over_without_an_attempt() -> (
    None
):
    """A concurrent request can make a deployment rest after this one has chosen its pool."""
    b, _ = board()
    victims: list[Deployment] = []

    class Rattler(FakeProvider):
        async def chat(self, request: ChatCompletionRequest) -> Any:
            limited(b, victims[0], retry_after=30)  # someone else's call gets a 429 meanwhile
            return await super().chat(request)

    failing = Rattler(fail_first=1)
    skipped, last = FakeProvider(), FakeProvider(reply="ok")
    first = dep(provider=failing, vendor="a")
    second, third = dep(provider=skipped, vendor="b"), dep(provider=last, vendor="c")
    victims.append(second)
    r, _ = router_for([first, second, third], b, attempts=2)

    response = await r.complete(request())

    assert response.choices[0].message.content == "ok"  # the refusal was not one of the 2 attempts
    assert skipped.calls == 0
