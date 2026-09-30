import random
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from app.providers.errors import (
    AuthFailed,
    BadRequest,
    ProviderDown,
    ProviderError,
    RateLimited,
    Timeout,
)
from app.providers.fake import FakeProvider
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry
from app.router.retry import RetryPolicy
from app.router.router import Router
from app.schemas.chat import ChatCompletionRequest, Message
from app.schemas.completion import ChatCompletionChunk
from tests.conftest import TEST_ALIAS, model_config


class FirstChoice(random.Random):
    """Always picks the first candidate and the top of any jitter range: fully deterministic."""

    def choices(  # type: ignore[override]
        self, population: Sequence[Any], weights: Any = None, **kwargs: Any
    ) -> list[Any]:
        return [population[0]]

    def uniform(self, a: float, b: float) -> float:
        return b


class Clock:
    def __init__(self) -> None:
        self.sleeps: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


POLICY = RetryPolicy(max_attempts=3, base_delay=0.5, multiplier=2.0, max_delay=8.0)
REQUEST = ChatCompletionRequest(model=TEST_ALIAS, messages=[Message(role="user", content="hi")])


def router_for(*providers: FakeProvider, policy: RetryPolicy = POLICY) -> tuple[Router, Clock]:
    clock = Clock()
    registry = ModelRegistry(
        Deployment.create(model_config(model=f"model-{i}"), provider)
        for i, provider in enumerate(providers)
    )
    return Router(registry, policy=policy, rng=FirstChoice(), sleep=clock.sleep), clock


# ---------- single deployment ----------


async def test_fails_twice_then_succeeds() -> None:
    provider = FakeProvider(fail_first=2)
    router, clock = router_for(provider)

    response = await router.complete(REQUEST)

    assert response.choices[0].message.content
    assert provider.calls == 3
    assert clock.sleeps == [0.5, 1.0]  # exponential backoff between the tries


async def test_gives_up_after_max_attempts_with_the_last_error() -> None:
    provider = FakeProvider(fail_first=99)
    router, clock = router_for(provider)

    with pytest.raises(ProviderDown):
        await router.complete(REQUEST)

    assert provider.calls == 3
    assert len(clock.sleeps) == 2  # no sleep after the final failure


async def test_bad_request_is_never_retried() -> None:
    provider = FakeProvider(fail_first=99, error_factory=lambda: BadRequest("nope"))
    router, clock = router_for(provider)

    with pytest.raises(BadRequest):
        await router.complete(REQUEST)

    assert provider.calls == 1
    assert clock.sleeps == []


async def test_max_attempts_of_one_disables_retrying() -> None:
    provider = FakeProvider(fail_first=1)
    router, _ = router_for(provider, policy=RetryPolicy(max_attempts=1))

    with pytest.raises(ProviderDown):
        await router.complete(REQUEST)

    assert provider.calls == 1


async def test_timeout_is_retried() -> None:
    provider = FakeProvider(fail_first=1, error_factory=lambda: Timeout("slow"))
    router, _ = router_for(provider)

    await router.complete(REQUEST)

    assert provider.calls == 2


async def test_rate_limit_on_the_only_deployment_waits_for_retry_after() -> None:
    provider = FakeProvider(
        fail_first=1, error_factory=lambda: RateLimited("slow down", retry_after=3.0)
    )
    router, clock = router_for(provider)

    await router.complete(REQUEST)

    assert provider.calls == 2
    assert clock.sleeps == [3.0]


async def test_a_retry_after_longer_than_max_delay_is_passed_on_not_cut_short() -> None:
    provider = FakeProvider(fail_first=1, error_factory=lambda: RateLimited("x", retry_after=30))
    router, clock = router_for(provider)

    with pytest.raises(RateLimited) as caught:
        await router.complete(REQUEST)

    assert provider.calls == 1  # retrying after 8 s would have ignored the vendor's 30
    assert clock.sleeps == []
    assert caught.value.retry_after == 30


# ---------- several deployments ----------


async def test_retries_on_a_different_deployment_without_waiting() -> None:
    broken = FakeProvider("broken", fail_first=99)
    healthy = FakeProvider("healthy")
    router, clock = router_for(broken, healthy)

    response = await router.complete(REQUEST)

    assert response.choices[0].message.content
    assert (broken.calls, healthy.calls) == (1, 1)
    assert clock.sleeps == []
    assert healthy.requests[0].model == "model-1"


@pytest.mark.parametrize(
    "error",
    [RateLimited("throttled"), AuthFailed("bad key"), Timeout("slow"), ProviderDown("down")],
)
async def test_every_retryable_failure_moves_to_the_next_deployment(error: ProviderError) -> None:
    broken = FakeProvider("broken", fail_first=99, error_factory=lambda: error)
    healthy = FakeProvider("healthy")
    router, clock = router_for(broken, healthy)

    await router.complete(REQUEST)

    assert healthy.calls == 1
    assert clock.sleeps == []


async def test_bad_request_does_not_try_other_deployments() -> None:
    first = FakeProvider("first", fail_first=99, error_factory=lambda: BadRequest("nope"))
    second = FakeProvider("second")
    router, _ = router_for(first, second)

    with pytest.raises(BadRequest):
        await router.complete(REQUEST)

    assert second.calls == 0


async def test_tries_each_deployment_before_reusing_one() -> None:
    a, b, c = (FakeProvider(n, fail_first=99) for n in "abc")
    router, clock = router_for(a, b, c)

    with pytest.raises(ProviderDown):
        await router.complete(REQUEST)

    assert (a.calls, b.calls, c.calls) == (1, 1, 1)
    assert clock.sleeps == []


async def test_waits_only_once_every_deployment_has_been_tried() -> None:
    a, b = FakeProvider("a", fail_first=99), FakeProvider("b", fail_first=99)
    router, clock = router_for(
        a, b, policy=RetryPolicy(max_attempts=4, base_delay=0.5, multiplier=2.0, max_delay=8.0)
    )

    with pytest.raises(ProviderDown):
        await router.complete(REQUEST)

    # attempt 1 -> a, fails; b is untried, so no wait
    # attempt 2 -> b, fails; everything has been tried, so back off (retry #2 -> 1.0s)
    # attempt 3 -> a again, fails; back off (retry #3 -> 2.0s)
    # attempt 4 -> a again, fails; out of attempts, raise without sleeping
    assert (a.calls, b.calls) == (3, 1)
    assert clock.sleeps == [1.0, 2.0]


# ---------- streaming ----------


async def collect(stream: AsyncIterator[ChatCompletionChunk]) -> list[ChatCompletionChunk]:
    return [chunk async for chunk in stream]


async def test_stream_retries_when_it_fails_before_the_first_chunk() -> None:
    broken = FakeProvider("broken", fail_first=99)
    healthy = FakeProvider("healthy", reply="a b")
    router, _ = router_for(broken, healthy)

    chunks = await collect(router.stream(REQUEST))

    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "a b"
    assert (broken.calls, healthy.calls) == (1, 1)


async def test_stream_gives_up_after_max_attempts() -> None:
    provider = FakeProvider(fail_first=99)
    router, _ = router_for(provider)

    with pytest.raises(ProviderDown):
        await collect(router.stream(REQUEST))

    assert provider.calls == 3


class DiesAfterFirstChunk(FakeProvider):
    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        self.calls += 1
        async for chunk in FakeProvider(reply="a b c").stream(request):
            yield chunk
            raise ProviderDown("reset", provider="fake")


async def test_stream_failure_after_output_started_is_not_retried() -> None:
    dying, healthy = DiesAfterFirstChunk("dying"), FakeProvider("healthy")
    router, _ = router_for(dying, healthy)
    received: list[ChatCompletionChunk] = []

    with pytest.raises(ProviderDown):
        async for chunk in router.stream(REQUEST):
            received.append(chunk)

    assert len(received) == 1
    assert healthy.calls == 0
    assert dying.calls == 1
