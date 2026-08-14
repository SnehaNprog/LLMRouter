import random

import pytest

from app.providers.errors import ProviderDown
from app.providers.fake import FakeProvider
from app.router.deployment import Deployment
from app.router.latency import LatencyTracker
from app.router.registry import ModelRegistry
from app.router.retry import RetryPolicy
from app.router.router import Router
from app.schemas.chat import ChatCompletionRequest, Message
from tests.conftest import TEST_ALIAS, model_config

# ---------- tracker ----------


def test_unknown_deployment_has_no_estimate() -> None:
    tracker = LatencyTracker()

    assert tracker.ewma("dep_x") is None
    assert tracker.stats("dep_x") is None


def test_first_observation_becomes_the_estimate() -> None:
    tracker = LatencyTracker(alpha=0.2)

    tracker.observe("a", 1.5)

    assert tracker.ewma("a") == 1.5
    assert tracker.stats("a") is not None


def test_ewma_follows_the_formula() -> None:
    tracker = LatencyTracker(alpha=0.2)

    for value in (1.0, 2.0, 3.0):
        tracker.observe("a", value)

    first = 1.0
    second = 0.2 * 2.0 + 0.8 * first
    third = 0.2 * 3.0 + 0.8 * second
    assert tracker.ewma("a") == pytest.approx(third)


def test_alpha_of_one_tracks_the_latest_value() -> None:
    tracker = LatencyTracker(alpha=1.0)

    tracker.observe("a", 5.0)
    tracker.observe("a", 0.5)

    assert tracker.ewma("a") == 0.5


def test_converges_to_a_steady_latency() -> None:
    tracker = LatencyTracker(alpha=0.2)
    tracker.observe("a", 10.0)

    for _ in range(60):
        tracker.observe("a", 1.0)

    assert tracker.ewma("a") == pytest.approx(1.0, abs=0.001)


def test_one_outlier_moves_the_estimate_only_a_little() -> None:
    tracker = LatencyTracker(alpha=0.2)
    for _ in range(20):
        tracker.observe("a", 1.0)

    tracker.observe("a", 11.0)

    assert tracker.ewma("a") == pytest.approx(3.0)  # 1 + 0.2 * (11 - 1)


def test_recovers_after_a_slow_period() -> None:
    tracker = LatencyTracker(alpha=0.3)
    for _ in range(20):
        tracker.observe("a", 5.0)
    slow = tracker.ewma("a")
    assert slow is not None

    for _ in range(20):
        tracker.observe("a", 0.5)

    recovered = tracker.ewma("a")
    assert recovered is not None
    assert recovered < slow / 5


def test_deployments_are_tracked_independently() -> None:
    tracker = LatencyTracker()

    tracker.observe("fast", 0.2)
    tracker.observe("slow", 4.0)
    tracker.observe("fast", 0.2)

    assert tracker.ewma("fast") == pytest.approx(0.2)
    assert tracker.ewma("slow") == pytest.approx(4.0)


def test_stats_count_samples_and_remember_the_last_value() -> None:
    tracker = LatencyTracker()
    for value in (1.0, 2.0, 3.0):
        tracker.observe("a", value)

    stats = tracker.stats("a")

    assert stats is not None
    assert (stats.samples, stats.last) == (3, 3.0)


def test_snapshot_is_a_copy() -> None:
    tracker = LatencyTracker()
    tracker.observe("a", 1.0)

    snapshot = tracker.snapshot()
    snapshot.clear()

    assert tracker.ewma("a") == 1.0


def test_negative_latency_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        LatencyTracker().observe("a", -0.1)


@pytest.mark.parametrize("alpha", [0, -0.5, 1.5])
def test_invalid_alpha_is_rejected(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        LatencyTracker(alpha=alpha)


# ---------- recorded by the router ----------


class FakeClock:
    """The provider's simulated latency advances this clock, so timings are exact."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


REQUEST = ChatCompletionRequest(model=TEST_ALIAS, messages=[Message(role="user", content="hi")])


def slow_provider(clock: FakeClock, latency: float, **kwargs: object) -> FakeProvider:
    return FakeProvider(latency=latency, sleep=clock.sleep, **kwargs)  # type: ignore[arg-type]


def make_router(
    clock: FakeClock, *deployments: Deployment, fallbacks: dict[str, list[str]] | None = None
) -> Router:
    return Router(
        ModelRegistry(deployments, fallbacks),
        policy=RetryPolicy(max_attempts=2, base_delay=0.0),
        rng=random.Random(0),
        sleep=clock.sleep,
        clock=clock,
        latency=LatencyTracker(alpha=0.5),
    )


async def test_router_records_the_latency_of_a_successful_call() -> None:
    clock = FakeClock()
    deployment = Deployment.create(model_config(), slow_provider(clock, 0.4))
    router = make_router(clock, deployment)

    await router.complete(REQUEST)

    assert router.latency.ewma(deployment.id) == pytest.approx(0.4)


async def test_router_smooths_across_requests() -> None:
    clock = FakeClock()
    provider = slow_provider(clock, 1.0)
    deployment = Deployment.create(model_config(), provider)
    router = make_router(clock, deployment)

    await router.complete(REQUEST)
    provider.latency = 3.0
    await router.complete(REQUEST)

    assert router.latency.ewma(deployment.id) == pytest.approx(2.0)  # 0.5*3 + 0.5*1


async def test_failures_are_not_recorded() -> None:
    clock = FakeClock()
    broken = Deployment.create(
        model_config(name="broken", model="b"), slow_provider(clock, 0.01, fail_first=99)
    )
    router = make_router(clock, broken)

    with pytest.raises(ProviderDown):
        await router.complete(
            ChatCompletionRequest(model="broken", messages=[Message(role="user", content="hi")])
        )

    assert router.latency.ewma(broken.id) is None


async def test_only_the_successful_retry_is_recorded() -> None:
    clock = FakeClock()
    flaky = Deployment.create(model_config(), slow_provider(clock, 0.7, fail_first=1))
    router = make_router(clock, flaky)

    await router.complete(REQUEST)

    stats = router.latency.stats(flaky.id)
    assert stats is not None
    assert stats.samples == 1
    assert stats.ewma == pytest.approx(0.7)


async def test_each_deployment_gets_its_own_estimate() -> None:
    clock = FakeClock()
    quick = Deployment.create(model_config(model="quick"), slow_provider(clock, 0.1))
    sluggish = Deployment.create(model_config(model="sluggish"), slow_provider(clock, 2.0))
    router = make_router(clock, quick, sluggish)

    for _ in range(40):
        await router.complete(REQUEST)

    assert router.latency.ewma(quick.id) == pytest.approx(0.1)
    assert router.latency.ewma(sluggish.id) == pytest.approx(2.0)


async def test_fallback_records_the_deployment_that_answered() -> None:
    clock = FakeClock()
    primary = Deployment.create(
        model_config(name="primary", model="p"), slow_provider(clock, 0.05, fail_first=99)
    )
    backup = Deployment.create(model_config(name="backup", model="b"), slow_provider(clock, 0.9))
    router = make_router(clock, primary, backup, fallbacks={"primary": ["backup"]})

    await router.complete(
        ChatCompletionRequest(model="primary", messages=[Message(role="user", content="hi")])
    )

    assert router.latency.ewma(primary.id) is None
    assert router.latency.ewma(backup.id) == pytest.approx(0.9)


async def test_streaming_records_time_to_first_chunk() -> None:
    clock = FakeClock()
    deployment = Deployment.create(model_config(), slow_provider(clock, 0.6, reply="a b c d"))
    router = make_router(clock, deployment)

    _ = [chunk async for chunk in router.stream(REQUEST)]

    assert router.latency.ewma(deployment.id) == pytest.approx(0.6)


async def test_stream_failing_before_first_chunk_is_not_recorded() -> None:
    clock = FakeClock()
    deployment = Deployment.create(model_config(), slow_provider(clock, 0.2, fail_first=99))
    router = make_router(clock, deployment)

    with pytest.raises(ProviderDown):
        _ = [chunk async for chunk in router.stream(REQUEST)]

    assert router.latency.ewma(deployment.id) is None
