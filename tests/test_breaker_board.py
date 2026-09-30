import asyncio
import random
from collections.abc import Sequence
from typing import Any

import httpx
import pytest

from app.core.config import Settings
from app.main import create_app
from app.providers.errors import AuthFailed, BadRequest, ProviderDown, RateLimited, Timeout
from app.providers.fake import FakeProvider
from app.resilience.board import (
    BreakerBoard,
    BreakerSettings,
    DeploymentUnavailable,
    Health,
    health_of,
    provider_key,
)
from app.resilience.breaker import CircuitState
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry
from app.router.retry import RetryPolicy
from app.router.router import Router
from app.schemas.chat import ChatCompletionRequest, Message
from tests.conftest import TEST_ALIAS, AuthedClient, model_config, registry_with

CLOSED, OPEN, HALF_OPEN = CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN
A_URL = "https://a.example.com/v1"
B_URL = "https://b.example.com/v1"
REQUEST = ChatCompletionRequest(model=TEST_ALIAS, messages=[Message(role="user", content="hi")])


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FirstChoice(random.Random):
    def choices(  # type: ignore[override]
        self, population: Sequence[Any], weights: Any = None, **kwargs: Any
    ) -> list[Any]:
        return [population[0]]

    def uniform(self, a: float, b: float) -> float:
        return b


def deployment(
    provider: FakeProvider | None = None,
    *,
    name: str = TEST_ALIAS,
    model: str = "m",
    url: str = A_URL,
) -> Deployment:
    return Deployment.create(
        model_config(name=name, model=model, base_url=url), provider or FakeProvider()
    )


def board_for(
    threshold: int = 3, cooldown: float = 30.0, **kwargs: Any
) -> tuple[BreakerBoard, Clock]:
    clock = Clock()
    settings = BreakerSettings(
        failure_threshold=threshold, failure_rate_threshold=None, cooldown=cooldown, **kwargs
    )
    return BreakerBoard(settings, clock=clock), clock


def fail(board: BreakerBoard, target: Deployment, error: Exception, times: int = 1) -> None:
    for _ in range(times):
        board.acquire(target).failure(error)


DOWN = ProviderDown("down")


# ---------- provider keys ----------


def test_models_on_one_endpoint_share_a_provider_key() -> None:
    one, two = model_config(model="x", base_url=A_URL), model_config(model="y", base_url=A_URL)
    assert provider_key(one) == provider_key(two)


def test_different_endpoints_or_kinds_have_different_keys() -> None:
    base = model_config(base_url=A_URL)
    assert provider_key(base) != provider_key(model_config(base_url=B_URL))
    assert provider_key(base) != provider_key(model_config(provider="anthropic", base_url=A_URL))


def test_an_endpoint_left_to_the_default_still_has_a_key() -> None:
    assert provider_key(model_config(provider="anthropic", base_url=None)) == "anthropic@default"


# ---------- what counts as a failure ----------


@pytest.mark.parametrize(
    ("error", "health"),
    [
        (ProviderDown("x"), Health.PROVIDER_DOWN),
        (Timeout("x"), Health.PROVIDER_DOWN),
        (AuthFailed("x"), Health.DEPLOYMENT_BAD),
        (BadRequest("x"), Health.NEUTRAL),
        (RateLimited("x"), Health.NEUTRAL),
        (ValueError("a bug"), Health.NEUTRAL),
        (asyncio.CancelledError(), Health.NEUTRAL),
    ],
)
def test_health_of_an_error(error: BaseException, health: Health) -> None:
    assert health_of(error) is health


# ---------- the board ----------


def test_a_deployment_opens_after_enough_failures() -> None:
    board, _ = board_for(threshold=3)
    target = deployment()
    fail(board, target, DOWN, 2)
    assert not board.is_open(target)
    fail(board, target, DOWN)
    assert board.is_open(target)
    with pytest.raises(DeploymentUnavailable) as caught:
        board.acquire(target)
    assert caught.value.retry_after == pytest.approx(30)
    assert caught.value.breaker.startswith("deployment:")


def test_a_provider_outage_opens_for_every_deployment_on_it() -> None:
    board, _ = board_for(threshold=3)
    one, two = deployment(model="one"), deployment(model="two")
    fail(board, one, DOWN, 3)
    assert board.provider_breaker(one).state is OPEN
    assert board.deployment_breaker(two).state is CLOSED  # two has never failed itself ...
    with pytest.raises(DeploymentUnavailable) as caught:
        board.acquire(two)  # ... but its vendor is down
    assert caught.value.breaker.startswith("provider:")


def test_failures_spread_over_deployments_add_up_at_the_provider() -> None:
    board, _ = board_for(threshold=4)
    one, two = deployment(model="one"), deployment(model="two")
    fail(board, one, DOWN, 2)
    fail(board, two, DOWN, 2)
    assert board.deployment_breaker(one).state is CLOSED
    assert board.deployment_breaker(two).state is CLOSED
    assert board.provider_breaker(one).state is OPEN


def test_another_vendor_is_unaffected() -> None:
    board, _ = board_for(threshold=2)
    fail(board, deployment(url=A_URL), DOWN, 2)
    board.acquire(deployment(url=B_URL)).success()


def test_a_rejected_key_opens_only_the_deployment() -> None:
    board, _ = board_for(threshold=2)
    bad, good = deployment(model="bad"), deployment(model="good")
    fail(board, bad, AuthFailed("no"), 2)
    assert board.deployment_breaker(bad).state is OPEN
    assert board.provider_breaker(bad).state is CLOSED
    board.acquire(good).success()  # the same vendor, with a working key, is still used


@pytest.mark.parametrize("error", [BadRequest("x"), RateLimited("x"), ValueError("bug")])
def test_failures_that_say_nothing_about_health_never_open_a_breaker(error: Exception) -> None:
    board, _ = board_for(threshold=2)
    target = deployment()
    fail(board, target, error, 20)
    assert not board.is_open(target)
    assert board.deployment_breaker(target).consecutive_failures == 0


def test_a_success_between_failures_keeps_it_closed() -> None:
    board, _ = board_for(threshold=3)
    target = deployment()
    for _ in range(10):
        fail(board, target, DOWN, 2)
        board.acquire(target).success()
    assert not board.is_open(target)


def test_after_the_cooldown_one_trial_call_decides() -> None:
    board, clock = board_for(threshold=2, cooldown=30)
    target = deployment()
    fail(board, target, DOWN, 2)
    clock.advance(30)
    trial = board.acquire(target)
    with pytest.raises(DeploymentUnavailable):
        board.acquire(target)  # only one trial at a time
    trial.success()
    assert not board.is_open(target)
    board.acquire(target).success()


def test_a_failed_trial_reopens_both() -> None:
    board, clock = board_for(threshold=2, cooldown=30)
    target = deployment()
    fail(board, target, DOWN, 2)
    clock.advance(30)
    board.acquire(target).failure(DOWN)
    assert board.deployment_breaker(target).state is OPEN
    assert board.provider_breaker(target).state is OPEN


def test_a_refusal_by_the_provider_does_not_leak_the_deployments_trial_slot() -> None:
    board, clock = board_for(threshold=2, cooldown=30)
    one, two = deployment(model="one"), deployment(model="two")
    fail(board, one, AuthFailed("no"), 2)  # one's own breaker opens
    fail(board, two, DOWN, 2)  # the provider's breaker opens (and two's)
    clock.advance(30)
    assert board.deployment_breaker(one).state is HALF_OPEN
    trial = board.acquire(two)  # takes the provider's only trial slot
    with pytest.raises(DeploymentUnavailable):
        board.acquire(one)  # one's own slot was free, but the provider has none left
    assert board.deployment_breaker(one).probes_in_flight == 0
    trial.success()
    board.acquire(one).success()


def test_releasing_a_permit_counts_for_nothing() -> None:
    board, _ = board_for(threshold=1)
    target = deployment()
    for _ in range(5):
        board.acquire(target).release()
    assert not board.is_open(target)


def test_a_permit_reports_only_once() -> None:
    board, _ = board_for(threshold=2)
    target = deployment()
    permit = board.acquire(target)
    permit.failure(DOWN)
    permit.failure(DOWN)
    assert board.deployment_breaker(target).consecutive_failures == 1


def test_states_lists_every_breaker_made() -> None:
    board, _ = board_for(threshold=1)
    target = deployment()
    fail(board, target, DOWN)
    assert sorted(board.states().values()) == ["open", "open"]
    assert any(name.startswith("provider:") for name in board.states())


def test_breakers_are_made_once_per_target() -> None:
    board, _ = board_for()
    target = deployment()
    assert board.deployment_breaker(target) is board.deployment_breaker(target)
    assert board.provider_breaker(target) is board.provider_breaker(deployment(model="other"))


def test_transitions_reach_the_observer() -> None:
    seen: list[tuple[str, str]] = []
    board = BreakerBoard(
        BreakerSettings(failure_threshold=1, failure_rate_threshold=None),
        on_transition=lambda name, old, new: seen.append((name.split(":")[0], new.value)),
    )
    fail(board, deployment(), DOWN)
    assert sorted(seen) == [("deployment", "open"), ("provider", "open")]


# ---------- in the router ----------


class Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.calls.append(seconds)


def router_for(
    *deployments: Deployment,
    threshold: int = 2,
    attempts: int = 3,
    fallbacks: dict[str, Sequence[str]] | None = None,
) -> tuple[Router, BreakerBoard, Clock, Sleeps]:
    board, clock = board_for(threshold=threshold)
    sleeps = Sleeps()
    router = Router(
        ModelRegistry(deployments, fallbacks),
        policy=RetryPolicy(max_attempts=attempts, base_delay=0.5),
        rng=FirstChoice(),
        sleep=sleeps.sleep,
        breakers=board,
    )
    return router, board, clock, sleeps


async def test_a_failing_deployment_is_left_alone_once_its_breaker_opens() -> None:
    bad, good = FakeProvider(fail_first=999), FakeProvider(reply="ok")
    router, _, _, _ = router_for(
        deployment(bad, model="bad", url=A_URL), deployment(good, model="good", url=B_URL)
    )

    for _ in range(6):
        assert (await router.complete(REQUEST)).choices[0].message.content == "ok"

    assert bad.calls == 2  # it failed twice, then was never called again
    assert good.calls == 6


async def test_a_vendor_outage_is_left_alone_across_its_models() -> None:
    one, two = FakeProvider(fail_first=999), FakeProvider(fail_first=999)
    elsewhere = FakeProvider(reply="ok")
    router, _, _, _ = router_for(
        deployment(one, model="one", url=A_URL),
        deployment(two, model="two", url=A_URL),
        deployment(elsewhere, model="three", url=B_URL),
        threshold=2,
    )

    await router.complete(REQUEST)  # one fails, two fails: the vendor is now open
    for _ in range(5):
        await router.complete(REQUEST)

    assert (one.calls, two.calls) == (1, 1)
    assert elsewhere.calls == 6


async def test_refused_deployments_neither_use_up_attempts_nor_wait() -> None:
    first, second = FakeProvider(), FakeProvider()
    last = FakeProvider(reply="finally")
    router, board, _, sleeps = router_for(
        deployment(first, model="a", url="https://one.example.com/v1"),
        deployment(second, model="b", url="https://two.example.com/v1"),
        deployment(last, model="c", url="https://three.example.com/v1"),
        attempts=1,  # a single attempt: the refusals must not use it up
    )
    for target in router.registry.get(TEST_ALIAS)[:2]:
        fail(board, target, DOWN, 2)

    response = await router.complete(REQUEST)

    assert response.choices[0].message.content == "finally"
    assert (first.calls, second.calls) == (0, 0)
    assert sleeps.calls == []


async def test_when_everything_is_open_the_request_fails_fast() -> None:
    provider = FakeProvider()
    router, board, _, sleeps = router_for(deployment(provider), threshold=1)
    fail(board, router.registry.get(TEST_ALIAS)[0], DOWN)

    with pytest.raises(DeploymentUnavailable):
        await router.complete(REQUEST)

    assert provider.calls == 0
    assert sleeps.calls == []  # no backoff for something that cannot succeed by waiting


async def test_it_falls_back_to_another_model_when_the_first_is_open() -> None:
    primary, backup = FakeProvider(), FakeProvider(reply="from backup")
    router, board, _, _ = router_for(
        deployment(primary, name="primary", url=A_URL),
        deployment(backup, name="backup", url=B_URL),
        fallbacks={"primary": ["backup"]},
        threshold=1,
    )
    fail(board, router.registry.get("primary")[0], DOWN)

    response = await router.complete(REQUEST.model_copy(update={"model": "primary"}))

    assert response.choices[0].message.content == "from backup"
    assert primary.calls == 0


async def test_a_bad_request_does_not_count_against_the_deployment() -> None:
    provider = FakeProvider(error_factory=lambda: BadRequest("nope"), fail_first=999)
    router, board, _, _ = router_for(deployment(provider), threshold=1)

    for _ in range(5):
        with pytest.raises(BadRequest):
            await router.complete(REQUEST)

    assert not board.is_open(router.registry.get(TEST_ALIAS)[0])
    assert provider.calls == 5


async def test_it_recovers_after_the_cooldown() -> None:
    provider = FakeProvider(fail_first=2, reply="back")
    router, board, clock, _ = router_for(deployment(provider), threshold=2, attempts=1)

    for _ in range(2):
        with pytest.raises(ProviderDown):
            await router.complete(REQUEST)
    with pytest.raises(DeploymentUnavailable):
        await router.complete(REQUEST)
    clock.advance(30)
    response = await router.complete(REQUEST)

    assert response.choices[0].message.content == "back"
    assert not board.is_open(router.registry.get(TEST_ALIAS)[0])


async def test_a_cancelled_call_frees_its_trial_slot() -> None:
    gate = asyncio.Event()
    started = asyncio.Event()

    class Hanging(FakeProvider):
        async def chat(self, request: ChatCompletionRequest) -> Any:
            started.set()
            await gate.wait()
            return await super().chat(request)

    provider = Hanging()
    router, board, clock, _ = router_for(deployment(provider), threshold=1, attempts=1)
    target = router.registry.get(TEST_ALIAS)[0]
    fail(board, target, DOWN)
    clock.advance(30)  # half-open: one trial allowed

    task = asyncio.create_task(router.complete(REQUEST))
    await started.wait()
    assert board.deployment_breaker(target).probes_in_flight == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert board.deployment_breaker(target).probes_in_flight == 0
    assert board.deployment_breaker(target).state is HALF_OPEN  # nothing learned, try again


async def test_streams_failing_before_the_first_chunk_count() -> None:
    provider = FakeProvider(fail_first=999)
    router, board, _, _ = router_for(deployment(provider), threshold=2, attempts=1)
    target = router.registry.get(TEST_ALIAS)[0]

    for _ in range(2):
        with pytest.raises(ProviderDown):
            [chunk async for chunk in router.stream(REQUEST)]

    assert board.is_open(target)
    with pytest.raises(DeploymentUnavailable):
        [chunk async for chunk in router.stream(REQUEST)]


async def test_a_stream_that_starts_counts_as_a_success() -> None:
    provider = FakeProvider(reply="a b c")
    router, board, _, _ = router_for(deployment(provider), threshold=2)
    target = router.registry.get(TEST_ALIAS)[0]
    fail(board, target, DOWN)

    chunks = [chunk async for chunk in router.stream(REQUEST)]

    assert chunks
    assert board.deployment_breaker(target).consecutive_failures == 0


async def test_the_default_router_has_working_breakers() -> None:
    router = Router(ModelRegistry([deployment(FakeProvider(fail_first=999))]), sleep=Sleeps().sleep)
    target = router.registry.get(TEST_ALIAS)[0]

    for _ in range(5):
        with pytest.raises(ProviderDown):
            await router.complete(REQUEST.model_copy())
        if router.breakers.is_open(target):
            break

    assert router.breakers.is_open(target)


# ---------- settings and the app ----------


def api_client(provider: FakeProvider, **settings: Any) -> AuthedClient:
    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return registry_with(provider)

    options = {"retry_base_delay": 0.0, "retry_max_attempts": 1, **settings}
    return AuthedClient(create_app(Settings(_env_file=None, **options), factory))


def ask(client: AuthedClient) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        json={"model": TEST_ALIAS, "messages": [{"role": "user", "content": "hi"}]},
    )


def test_the_app_uses_the_breaker_settings() -> None:
    provider = FakeProvider(fail_first=999)
    with api_client(provider, breaker_failure_threshold=2) as client:
        statuses = [ask(client).status_code for _ in range(4)]

    assert statuses == [502, 502, 503, 503]  # then every provider is behind an open breaker
    assert provider.calls == 2  # the breaker opened after two failures; the rest never got through


def test_a_rate_only_setup_works() -> None:
    provider = FakeProvider(fail_first=999)
    with api_client(
        provider,
        breaker_failure_threshold=0,
        breaker_failure_rate=0.5,
        breaker_min_requests=4,
    ) as client:
        for _ in range(8):
            ask(client)

    assert provider.calls == 4


def test_a_refusal_by_the_breakers_is_a_503_with_a_retry_time() -> None:
    with api_client(FakeProvider(fail_first=999), breaker_failure_threshold=1) as client:
        ask(client)
        response = ask(client)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "providers_unavailable"
    assert 1 <= int(response.headers["Retry-After"]) <= 30


def test_both_rules_off_is_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="breaker_failure_threshold"):
        Settings(_env_file=None, breaker_failure_threshold=0, breaker_failure_rate=0)


@pytest.mark.parametrize(
    "field", ["breaker_failure_rate", "breaker_min_requests", "breaker_window_seconds"]
)
def test_nonsense_breaker_settings_are_rejected(field: str) -> None:
    bad: dict[str, Any] = {
        "breaker_failure_rate": 1.5,
        "breaker_min_requests": 0,
        "breaker_window_seconds": 0,
    }
    with pytest.raises(ValueError):
        Settings(_env_file=None, **{field: bad[field]})


async def test_a_refusal_does_not_count_towards_the_attempt_limit() -> None:
    refused, broken, healthy = FakeProvider(), FakeProvider(fail_first=1), FakeProvider(reply="ok")
    router, board, _, _ = router_for(
        deployment(refused, model="a", url="https://one.example.com/v1"),
        deployment(broken, model="b", url="https://two.example.com/v1"),
        deployment(healthy, model="c", url="https://three.example.com/v1"),
        attempts=2,  # one real failure is allowed to be followed by one more real try
    )
    fail(board, router.registry.get(TEST_ALIAS)[0], DOWN, 2)

    response = await router.complete(REQUEST)

    assert response.choices[0].message.content == "ok"
    assert (refused.calls, broken.calls, healthy.calls) == (0, 1, 1)
