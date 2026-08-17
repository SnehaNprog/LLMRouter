import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.model_config import ConfigError, load_config, parse_config
from app.main import create_app
from app.providers.errors import ProviderDown
from app.providers.fake import FakeProvider
from app.router.context import ContextLengthExceededError, fits
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry
from app.router.retry import RetryPolicy
from app.router.router import Router
from app.schemas.chat import ChatCompletionRequest, Message
from tests.conftest import model_config

SHORT = "hi"  # about 8 prompt tokens
LONG = "a" * 800  # about 207 prompt tokens


def request(content: str = SHORT, model: str = "chat", **extra: Any) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model, messages=[Message(role="user", content=content)], **extra
    )


def deployment(
    window: int | None, alias: str = "chat", provider: FakeProvider | None = None
) -> Deployment:
    return Deployment.create(
        model_config(name=alias, model=f"{alias}-{window}", context_window=window),
        provider or FakeProvider(f"{alias}-{window}"),
    )


class FirstChoice(random.Random):
    def choices(  # type: ignore[override]
        self, population: Sequence[Any], weights: Any = None, **kwargs: Any
    ) -> list[Any]:
        return [population[0]]

    def uniform(self, a: float, b: float) -> float:
        return b


def router_for(*deployments: Deployment, fallbacks: dict[str, list[str]] | None = None) -> Router:
    async def no_sleep(_: float) -> None:
        return None

    return Router(
        ModelRegistry(deployments, fallbacks),
        policy=RetryPolicy(max_attempts=2, base_delay=0.0),
        rng=FirstChoice(),
        sleep=no_sleep,
    )


# ---------- config ----------


def entry(**extra: Any) -> dict[str, Any]:
    return {
        "models": [
            {
                "name": "a",
                "provider": "openai_compatible",
                "model": "m",
                "base_url": "https://api.example.com/v1",
                **extra,
            }
        ]
    }


def test_context_window_is_read_from_config() -> None:
    assert parse_config(entry(context_window=8192), {}).models[0].context_window == 8192


def test_context_window_is_optional() -> None:
    assert parse_config(entry(), {}).models[0].context_window is None


@pytest.mark.parametrize("value", [0, -5])
def test_non_positive_context_window_is_rejected(value: int) -> None:
    with pytest.raises(ConfigError, match="context_window"):
        parse_config(entry(context_window=value), {})


def test_context_window_does_not_change_the_deployment_id() -> None:
    small = Deployment.create(model_config(context_window=1000), FakeProvider())
    large = Deployment.create(model_config(context_window=9999), FakeProvider())

    assert small.id == large.id


def test_shipped_example_declares_windows() -> None:
    config = load_config(
        Path(__file__).parent.parent / "config.example.yaml",
        {"GROQ_API_KEY": "a", "ANTHROPIC_API_KEY": "b"},
    )

    assert all(m.context_window for m in config.models)


# ---------- fits ----------


def test_unknown_window_accepts_anything() -> None:
    assert fits(deployment(None), 10_000_000)


def test_boundary_is_inclusive() -> None:
    assert fits(deployment(100), 100)
    assert not fits(deployment(100), 101)


# ---------- within one alias ----------


async def test_long_prompt_goes_to_the_bigger_model() -> None:
    small, big = FakeProvider("small"), FakeProvider("big")
    router = router_for(deployment(100, provider=small), deployment(10_000, provider=big))

    await router.complete(request(LONG))

    assert (small.calls, big.calls) == (0, 1)


async def test_short_prompt_may_use_either_model() -> None:
    small, big = FakeProvider("small"), FakeProvider("big")
    router = router_for(deployment(100, provider=small), deployment(10_000, provider=big))

    await router.complete(request(SHORT))

    assert small.calls == 1  # the first candidate is eligible when the prompt is short


async def test_requested_completion_counts_towards_the_window() -> None:
    small, big = FakeProvider("small"), FakeProvider("big")
    router = router_for(deployment(100, provider=small), deployment(10_000, provider=big))

    await router.complete(request(SHORT, max_tokens=500))  # 8 + 500 > 100

    assert (small.calls, big.calls) == (0, 1)


async def test_deployment_without_a_window_is_always_eligible() -> None:
    unbounded = FakeProvider("unbounded")
    router = router_for(
        deployment(100, provider=FakeProvider("small")), deployment(None, provider=unbounded)
    )

    await router.complete(request(LONG))

    assert unbounded.calls == 1


async def test_retries_stay_among_models_that_fit() -> None:
    small = FakeProvider("small")
    flaky_big = FakeProvider("big", fail_first=1)
    router = router_for(deployment(100, provider=small), deployment(10_000, provider=flaky_big))

    await router.complete(request(LONG))

    assert small.calls == 0
    assert flaky_big.calls == 2


async def test_upstream_model_of_the_chosen_deployment_is_used() -> None:
    big = FakeProvider("big")
    router = router_for(deployment(100), deployment(10_000, provider=big))

    await router.complete(request(LONG))

    assert big.requests[0].model == "chat-10000"


# ---------- across fallbacks ----------


async def test_alias_that_cannot_fit_is_skipped_for_its_fallback() -> None:
    primary, backup = FakeProvider("primary"), FakeProvider("backup")
    router = router_for(
        deployment(100, alias="primary", provider=primary),
        deployment(10_000, alias="backup", provider=backup),
        fallbacks={"primary": ["backup"]},
    )

    await router.complete(request(LONG, model="primary"))

    assert (primary.calls, backup.calls) == (0, 1)


async def test_nothing_fits_raises_with_the_largest_window() -> None:
    router = router_for(
        deployment(100, alias="primary"),
        deployment(150, alias="backup"),
        fallbacks={"primary": ["backup"]},
    )

    with pytest.raises(ContextLengthExceededError) as excinfo:
        await router.complete(request(LONG, model="primary"))

    assert excinfo.value.window == 150
    assert excinfo.value.required > 150
    assert "maximum context length is 150 tokens" in str(excinfo.value)


async def test_real_failure_beats_a_context_error() -> None:
    primary = FakeProvider("primary", fail_first=99)
    router = router_for(
        deployment(10_000, alias="primary", provider=primary),
        deployment(100, alias="backup"),
        fallbacks={"primary": ["backup"]},
    )

    with pytest.raises(ProviderDown):  # the backup was too small, but primary really failed
        await router.complete(request(LONG, model="primary"))


async def test_unknown_model_still_wins_over_context_checks() -> None:
    from app.router.registry import UnknownModelError

    router = router_for(deployment(100))

    with pytest.raises(UnknownModelError):
        await router.complete(request(LONG, model="nope"))


# ---------- streaming ----------


async def test_stream_uses_a_model_that_fits() -> None:
    small, big = FakeProvider("small"), FakeProvider("big", reply="x y")
    router = router_for(deployment(100, provider=small), deployment(10_000, provider=big))

    chunks = [chunk async for chunk in router.stream(request(LONG))]

    assert small.calls == 0
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "x y"


async def test_stream_raises_before_yielding_when_nothing_fits() -> None:
    router = router_for(deployment(100))

    with pytest.raises(ContextLengthExceededError):
        async for _ in router.stream(request(LONG)):
            pytest.fail("no chunk should be produced")


# ---------- over HTTP ----------


def client_for(*deployments: Deployment) -> TestClient:
    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return ModelRegistry(deployments)

    return TestClient(create_app(Settings(_env_file=None, retry_base_delay=0.0), factory))


def test_too_long_request_is_an_openai_shaped_400() -> None:
    with client_for(deployment(100)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "chat", "messages": [{"role": "user", "content": LONG}]},
        )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "messages"
    assert "maximum context length is 100 tokens" in error["message"]


def test_too_long_streaming_request_is_a_normal_400() -> None:
    with client_for(deployment(100)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "chat", "stream": True, "messages": [{"role": "user", "content": LONG}]},
        )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")


def test_long_request_is_served_by_the_bigger_model_over_http() -> None:
    big = FakeProvider("big", reply="big answer")
    with client_for(deployment(100), deployment(10_000, provider=big)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "chat", "messages": [{"role": "user", "content": LONG}]},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "big answer"
