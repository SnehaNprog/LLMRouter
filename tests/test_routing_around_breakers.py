import random
from collections.abc import Sequence
from typing import Any

import fakeredis
import httpx
import pytest

from app.core.config import Settings
from app.core.tiers import Tier
from app.main import create_app
from app.providers.errors import AuthFailed, ProviderDown
from app.providers.fake import FakeProvider
from app.resilience.board import BreakerBoard, BreakerSettings, DeploymentUnavailable
from app.resilience.shared import SharedBreakerStore
from app.router.deployment import Deployment
from app.router.latency import LatencyTracker
from app.router.registry import ModelRegistry
from app.router.retry import RetryPolicy
from app.router.router import Router
from app.routing.resolve import resolve_model
from app.routing.score import ComplexityScore
from app.routing.selector import choose_model
from app.schemas.chat import ChatCompletionRequest
from tests.conftest import AuthedClient, model_config

DOWN = ProviderDown("down")
URLS = {n: f"https://{n}.example.com/v1" for n in "abcd"}


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Sleeps:
    async def sleep(self, seconds: float) -> None:
        pass


def dep(
    alias: str,
    vendor: str,
    *,
    tier: Tier | None = None,
    provider: FakeProvider | None = None,
    price: float = 0.0,
    weight: int = 1,
    model: str | None = None,
) -> Deployment:
    return Deployment.create(
        model_config(
            name=alias,
            model=model or f"{alias}-{vendor}",
            base_url=URLS[vendor],
            tier=tier,
            weight=weight,
            input_cost_per_mtok=price,
            output_cost_per_mtok=price,
        ),
        provider or FakeProvider(f"{alias}-{vendor}"),
    )


def board(threshold: int = 1, cooldown: float = 30.0) -> tuple[BreakerBoard, Clock]:
    clock = Clock()
    settings = BreakerSettings(
        failure_threshold=threshold, failure_rate_threshold=None, cooldown=cooldown
    )
    return BreakerBoard(settings, clock=clock), clock


def break_down(b: BreakerBoard, *targets: Deployment) -> None:
    for target in targets:
        b.acquire(target).failure(DOWN)


def request(model: str = "m", **extra: Any) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {"model": model, "messages": [{"role": "user", "content": "hi"}], **extra}
    )


def router(
    deployments: Sequence[Deployment],
    b: BreakerBoard,
    fallbacks: dict[str, Sequence[str]] | None = None,
    attempts: int = 3,
) -> Router:
    return Router(
        ModelRegistry(deployments, fallbacks),
        policy=RetryPolicy(max_attempts=attempts, base_delay=0.0),
        sleep=Sleeps().sleep,
        breakers=b,
        rng=random.Random(1),
    )


# ---------- the router picks among healthy deployments only ----------


async def test_an_open_deployment_is_never_picked_among_several() -> None:
    b, _ = board()
    bad, good_1, good_2 = FakeProvider(), FakeProvider(), FakeProvider()
    deployments = [
        dep("m", "a", provider=bad),
        dep("m", "b", provider=good_1),
        dep("m", "c", provider=good_2),
    ]
    r = router(deployments, b)
    break_down(b, deployments[0])

    for _ in range(40):
        await r.complete(request())

    assert bad.calls == 0
    assert good_1.calls > 0 and good_2.calls > 0  # the weights still apply among the healthy


async def test_picking_around_an_open_deployment_uses_no_attempt_on_it() -> None:
    """With one attempt allowed, the healthy deployment must still be the one that is tried."""
    b, _ = board()
    bad, good = FakeProvider(), FakeProvider(reply="ok")
    deployments = [dep("m", "a", provider=bad), dep("m", "b", provider=good)]
    r = router(deployments, b, attempts=1)
    break_down(b, deployments[0])

    for _ in range(20):
        assert (await r.complete(request())).choices[0].message.content == "ok"


async def test_a_vendor_outage_removes_all_its_models_from_the_pool() -> None:
    b, _ = board()
    one, two, other = FakeProvider(), FakeProvider(), FakeProvider(reply="ok")
    deployments = [
        dep("m", "a", provider=one, model="x"),
        dep("m", "a", provider=two, model="y"),
        dep("m", "b", provider=other),
    ]
    r = router(deployments, b)
    break_down(b, deployments[0])  # the vendor breaker opens, taking both of its models with it

    for _ in range(20):
        await r.complete(request())

    assert (one.calls, two.calls) == (0, 0)
    assert other.calls == 20


async def test_an_alias_with_nothing_healthy_falls_through_without_being_called() -> None:
    b, _ = board()
    primary, backup = FakeProvider(), FakeProvider(reply="from backup")
    deployments = [dep("primary", "a", provider=primary), dep("backup", "b", provider=backup)]
    r = router(deployments, b, fallbacks={"primary": ["backup"]})
    break_down(b, deployments[0])

    response = await r.complete(request("primary"))

    assert response.choices[0].message.content == "from backup"
    assert primary.calls == 0


async def test_a_half_open_deployment_is_given_its_trial_call() -> None:
    b, clock = board(cooldown=30)
    provider = FakeProvider(reply="back")
    deployments = [dep("m", "a", provider=provider)]
    r = router(deployments, b)
    break_down(b, deployments[0])
    clock.advance(30)

    response = await r.complete(request())

    assert response.choices[0].message.content == "back"
    assert not b.is_open(deployments[0])


# ---------- when everything is down ----------


async def test_with_everything_down_it_says_when_to_come_back() -> None:
    b, clock = board(cooldown=30)
    first, second = dep("m", "a"), dep("n", "b")
    r = router([first, second], b, fallbacks={"m": ["n"]})
    break_down(b, first)
    clock.advance(10)
    break_down(b, second)  # opened later: it will be available later

    with pytest.raises(DeploymentUnavailable) as caught:
        await r.complete(request("m"))

    assert caught.value.retry_after == pytest.approx(20)  # the soonest: first, 20 s from now
    assert caught.value.breaker == "all"


async def test_nothing_is_called_when_everything_is_down() -> None:
    b, _ = board()
    provider = FakeProvider()
    target = dep("m", "a", provider=provider)
    r = router([target], b)
    break_down(b, target)

    with pytest.raises(DeploymentUnavailable):
        await r.complete(request())

    assert provider.calls == 0


async def test_a_real_failure_is_reported_in_preference_to_the_outage() -> None:
    b, _ = board(threshold=100)
    r = router([dep("m", "a", provider=FakeProvider(fail_first=99))], b, attempts=1)

    with pytest.raises(ProviderDown) as caught:
        await r.complete(request())

    assert not isinstance(caught.value, DeploymentUnavailable)


async def test_streams_are_refused_before_any_output_when_everything_is_down() -> None:
    b, _ = board()
    target = dep("m", "a")
    r = router([target], b)
    break_down(b, target)

    with pytest.raises(DeploymentUnavailable):
        [chunk async for chunk in r.stream(request())]


# ---------- smart routing avoids what is down ----------


def ladder(b: BreakerBoard) -> list[Deployment]:
    return [
        dep("cheap", "a", tier=Tier.SLM, price=0.05),
        dep("middle", "b", tier=Tier.MID, price=0.6),
        dep("top", "c", tier=Tier.FRONTIER, price=3.0),
    ]


def choice_for(deployments: list[Deployment], b: BreakerBoard, complexity: float = 0.1) -> Any:
    registry = ModelRegistry(deployments)
    return choose_model(
        request("auto"),
        registry,
        LatencyTracker(),
        score=ComplexityScore(value=complexity, contributions={}, small_talk=False),
        healthy=lambda d: not b.is_open(d),
    )


def test_the_cheapest_model_is_skipped_while_its_vendor_is_down() -> None:
    b, _ = board()
    deployments = ladder(b)
    break_down(b, deployments[0])

    choice = choice_for(deployments, b, complexity=0.05)

    assert choice.alias == "middle"
    assert "cheap" not in [c.alias for c in choice.candidates]
    assert choice.unavailable == ["cheap"]
    assert any("circuit breaker open: cheap" in reason for reason in choice.reasons)


def test_the_choice_is_unchanged_while_everything_is_up() -> None:
    b, _ = board()
    choice = choice_for(ladder(b), b, complexity=0.05)
    assert choice.alias == "cheap"
    assert choice.unavailable == []


def test_a_down_model_is_not_offered_as_an_escalation_target() -> None:
    b, _ = board()
    deployments = ladder(b)
    break_down(b, deployments[1])

    choice = choice_for(deployments, b, complexity=0.05)

    assert choice.alias == "cheap"
    assert choice.escalation == ["top"]


def test_a_model_is_judged_on_its_healthy_deployments_only() -> None:
    b, _ = board()
    cheap_but_down = dep("m", "a", tier=Tier.MID, price=0.1, model="cheap-one")
    dear_but_up = dep("m", "b", tier=Tier.MID, price=10.0, model="dear-one")
    other = dep("n", "c", tier=Tier.MID, price=5.0)
    break_down(b, cheap_but_down)

    choice = choice_for([cheap_but_down, dear_but_up, other], b, complexity=0.4)

    prices = {c.alias: c.expected_cost_usd for c in choice.candidates}
    assert prices["m"] == pytest.approx(prices["n"] * 2)  # m costs what its one live one costs
    assert choice.alias == "n"


def test_when_everything_is_down_the_ranking_is_as_if_nothing_were() -> None:
    b, _ = board()
    deployments = ladder(b)
    break_down(b, *deployments)

    choice = choice_for(deployments, b, complexity=0.05)

    assert choice.alias == "cheap"
    assert choice.unavailable == []


def test_resolve_model_passes_the_health_check_through() -> None:
    b, _ = board()
    deployments = ladder(b)
    break_down(b, deployments[0])

    resolved, choice = resolve_model(
        request("auto"),
        ModelRegistry(deployments),
        LatencyTracker(),
        healthy=lambda d: not b.is_open(d),
    )

    assert choice is not None
    assert "cheap" not in [c.alias for c in choice.candidates]
    assert resolved.model == choice.alias


# ---------- through the API ----------


def api_client(deployments: list[Deployment], **settings: Any) -> AuthedClient:
    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return ModelRegistry(deployments)

    options = {"retry_base_delay": 0.0, "retry_max_attempts": 1, **settings}
    return AuthedClient(create_app(Settings(_env_file=None, **options), factory))


def post(client: AuthedClient, model: str, **extra: Any) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hi"}], **extra},
    )


def test_auto_stops_choosing_a_model_whose_vendor_keeps_failing() -> None:
    cheap_provider = FakeProvider(fail_first=999)
    deployments = [
        dep("cheap", "a", tier=Tier.SLM, provider=cheap_provider, price=0.05),
        dep("middle", "b", tier=Tier.MID, provider=FakeProvider(), price=0.6),
    ]
    with api_client(deployments, breaker_failure_threshold=2) as client:
        responses = [post(client, "auto") for _ in range(8)]

    assert cheap_provider.calls == 2  # it failed twice, and auto then routed around it
    later = responses[2:]
    assert {r.headers["x-llmrouter-model"] for r in later} == {"middle"}
    # and routed straight to it: it was chosen, not reached by escalating from the broken one
    assert {r.headers["x-llmrouter-escalated"] for r in later} == {"false"}


def test_a_total_outage_is_a_503_with_retry_after() -> None:
    deployments = [dep("m", "a", provider=FakeProvider(fail_first=999))]
    with api_client(deployments, breaker_failure_threshold=1) as client:
        first = post(client, "m")
        second = post(client, "m")

    assert first.status_code == 502
    assert second.status_code == 503
    assert second.json()["error"]["code"] == "providers_unavailable"
    assert 1 <= int(second.headers["Retry-After"]) <= 30


def test_a_streamed_request_in_a_total_outage_is_a_proper_503() -> None:
    deployments = [dep("m", "a", provider=FakeProvider(fail_first=999))]
    with api_client(deployments, breaker_failure_threshold=1) as client:
        post(client, "m")
        response = post(client, "m", stream=True)

    assert response.status_code == 503
    assert "Retry-After" in response.headers


def test_auto_in_a_total_outage_is_a_503_too() -> None:
    deployments = [
        dep("only", "a", tier=Tier.SLM, provider=FakeProvider(fail_first=999)),
    ]
    with api_client(deployments, breaker_failure_threshold=1) as client:
        post(client, "auto")
        response = post(client, "auto")

    assert response.status_code == 503


# ---------- what other replicas know ----------


class Replica:
    def __init__(self, redis: Any) -> None:
        self.clock = Clock()
        self.board = BreakerBoard(
            BreakerSettings(failure_threshold=1, failure_rate_threshold=None),
            clock=self.clock,
            shared=SharedBreakerStore(redis, "t:", clock=self.clock),
            sync_interval=0.0,
        )


async def test_a_total_outage_known_only_to_another_replica_is_still_reported_as_one() -> None:
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    one, two = Replica(redis), Replica(redis)
    target = dep("m", "a")
    break_down(one.board, target)
    await one.board.drain()

    provider = FakeProvider()
    r = router([dep("m", "a", provider=provider)], two.board)
    with pytest.raises(DeploymentUnavailable) as caught:
        await r.complete(request())

    assert caught.value.breaker == "all"
    assert provider.calls == 0


# ---------- how long to wait ----------


def test_the_wait_is_until_both_the_deployment_and_its_vendor_may_be_tried() -> None:
    b, clock = board(threshold=1, cooldown=30)
    first, second = dep("m", "a", model="x"), dep("m", "a", model="y")
    b.acquire(first).failure(AuthFailed("no"))  # only first's own breaker opens, now
    clock.advance(10)
    b.acquire(second).failure(DOWN)  # the vendor's breaker opens ten seconds later

    assert b.wait_for(first) == pytest.approx(30)  # the vendor's, the later of the two
    assert b.wait_for(dep("m", "b")) == 0


def test_auto_avoids_a_model_another_replica_has_found_down() -> None:
    cheap_provider = FakeProvider()
    cheap = dep("cheap", "a", tier=Tier.SLM, provider=cheap_provider, price=0.05)
    middle = dep("middle", "b", tier=Tier.MID, provider=FakeProvider(), price=0.6)
    with api_client([cheap, middle], breaker_sync_interval=0) as client:
        breakers = client.app.state.router.breakers  # type: ignore[attr-defined]

        async def another_replica_opens_cheap() -> None:
            redis = client.app.state.redis  # type: ignore[attr-defined]
            for name in (
                breakers.deployment_breaker(cheap).name,
                breakers.provider_breaker(cheap).name,
            ):
                await redis.set("llmrouter:breaker:" + name, "open", px=30_000)

        assert client.portal is not None
        client.portal.call(another_replica_opens_cheap)
        response = post(client, "auto")

    assert response.status_code == 200
    assert response.headers["x-llmrouter-model"] == "middle"
    assert response.headers["x-llmrouter-escalated"] == "false"  # chosen, not escalated to
    assert cheap_provider.calls == 0
