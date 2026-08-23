import random
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.core.config import Settings
from app.core.model_config import ConfigError, load_config, parse_config
from app.main import create_app
from app.providers.errors import BadRequest, ProviderDown, RateLimited, Timeout
from app.providers.fake import FakeProvider
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry, UnknownModelError
from app.router.retry import RetryPolicy
from app.router.router import Router
from app.schemas.chat import ChatCompletionRequest, Message
from app.schemas.completion import ChatCompletionChunk
from tests.conftest import AuthedClient, model_config

# ---------- config ----------


def entry(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "provider": "openai_compatible",
        "model": f"{name}-model",
        "base_url": "https://api.example.com/v1",
    }


def raw(fallbacks: object = None, names: Sequence[str] = ("a", "b", "c")) -> dict[str, Any]:
    data: dict[str, Any] = {"models": [entry(n) for n in names]}
    if fallbacks is not None:
        data["fallbacks"] = fallbacks
    return data


def test_fallbacks_are_optional() -> None:
    assert parse_config(raw(), {}).fallbacks == {}


def test_fallbacks_are_parsed_in_order() -> None:
    config = parse_config(raw({"a": ["b", "c"], "b": ["c"]}), {})

    assert config.fallbacks == {"a": ("b", "c"), "b": ("c",)}


def test_unknown_source_alias_is_rejected() -> None:
    with pytest.raises(ConfigError, match=r"fallbacks\.zzz: unknown model 'zzz'"):
        parse_config(raw({"zzz": ["a"]}), {})


def test_unknown_target_alias_is_rejected() -> None:
    with pytest.raises(ConfigError, match=r"fallbacks\.a\[1\]: unknown model 'nope'"):
        parse_config(raw({"a": ["b", "nope"]}), {})


def test_self_fallback_is_rejected() -> None:
    with pytest.raises(ConfigError, match="cannot fall back to itself"):
        parse_config(raw({"a": ["a"]}), {})


def test_fallback_problems_are_reported_with_other_problems() -> None:
    bad = raw({"a": ["nope"]})
    bad["models"][0]["api_key"] = "os.environ/MISSING_KEY"

    with pytest.raises(ConfigError) as excinfo:
        parse_config(bad, {})

    assert "MISSING_KEY" in str(excinfo.value)
    assert "unknown model 'nope'" in str(excinfo.value)


def test_shipped_example_has_a_valid_fallback() -> None:
    config = load_config(
        Path(__file__).parent.parent / "config.example.yaml",
        {"GROQ_API_KEY": "a", "ANTHROPIC_API_KEY": "b"},
    )

    assert config.fallbacks == {"fast": ("smart",)}


# ---------- registry chain ----------


def deployment(alias: str, provider: FakeProvider | None = None) -> Deployment:
    return Deployment.create(
        model_config(name=alias, model=f"{alias}-model"), provider or FakeProvider(alias)
    )


def registry(fallbacks: dict[str, list[str]], *names: str) -> ModelRegistry:
    return ModelRegistry([deployment(n) for n in names], fallbacks)


def test_chain_without_fallbacks_is_just_the_alias() -> None:
    assert registry({}, "a", "b").chain("a") == ["a"]


def test_chain_lists_fallbacks_in_order() -> None:
    assert registry({"a": ["b", "c"]}, "a", "b", "c").chain("a") == ["a", "b", "c"]


def test_chain_is_transitive() -> None:
    assert registry({"a": ["b"], "b": ["c"]}, "a", "b", "c").chain("a") == ["a", "b", "c"]


def test_chain_is_depth_first() -> None:
    chain = registry({"a": ["b", "d"], "b": ["c"]}, "a", "b", "c", "d").chain("a")

    assert chain == ["a", "b", "c", "d"]


def test_chain_survives_cycles() -> None:
    reg = registry({"a": ["b"], "b": ["a"]}, "a", "b")

    assert reg.chain("a") == ["a", "b"]
    assert reg.chain("b") == ["b", "a"]


def test_chain_does_not_repeat_aliases() -> None:
    assert registry({"a": ["b", "c"], "b": ["c"]}, "a", "b", "c").chain("a") == ["a", "b", "c"]


def test_chain_for_unknown_alias_raises() -> None:
    with pytest.raises(UnknownModelError):
        registry({}, "a").chain("nope")


def test_registry_rejects_fallbacks_to_unknown_models() -> None:
    with pytest.raises(ConfigError, match="unknown model 'ghost'"):
        registry({"a": ["ghost"]}, "a")


# ---------- router ----------


class FirstChoice(random.Random):
    def choices(  # type: ignore[override]
        self, population: Sequence[Any], weights: Any = None, **kwargs: Any
    ) -> list[Any]:
        return [population[0]]

    def uniform(self, a: float, b: float) -> float:
        return b


class Clock:
    async def sleep(self, seconds: float) -> None:
        return None


POLICY = RetryPolicy(max_attempts=2, base_delay=0.0)


def router_for(providers: dict[str, FakeProvider], fallbacks: dict[str, list[str]]) -> Router:
    reg = ModelRegistry([deployment(alias, p) for alias, p in providers.items()], fallbacks)
    return Router(reg, policy=POLICY, rng=FirstChoice(), sleep=Clock().sleep)


def request(model: str = "primary") -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model, messages=[Message(role="user", content="hi")])


async def test_dead_primary_falls_back_to_secondary() -> None:
    primary = FakeProvider("primary", fail_first=99)
    secondary = FakeProvider("secondary", reply="from secondary")
    router = router_for({"primary": primary, "secondary": secondary}, {"primary": ["secondary"]})

    response = await router.complete(request())

    assert response.choices[0].message.content == "from secondary"
    assert primary.calls == 2  # the primary is retried first
    assert secondary.calls == 1
    assert secondary.requests[0].model == "secondary-model"


async def test_healthy_primary_never_touches_the_fallback() -> None:
    primary, secondary = FakeProvider("primary"), FakeProvider("secondary")
    router = router_for({"primary": primary, "secondary": secondary}, {"primary": ["secondary"]})

    await router.complete(request())

    assert (primary.calls, secondary.calls) == (1, 0)


async def test_fallbacks_are_tried_in_order() -> None:
    dead = FakeProvider("primary", fail_first=99)
    second_dead = FakeProvider("second", fail_first=99)
    third = FakeProvider("third", reply="third wins")
    router = router_for(
        {"primary": dead, "second": second_dead, "third": third},
        {"primary": ["second", "third"]},
    )

    response = await router.complete(request())

    assert response.choices[0].message.content == "third wins"


async def test_transitive_chain_is_followed() -> None:
    a, b, c = FakeProvider("a", fail_first=99), FakeProvider("b", fail_first=99), FakeProvider("c")
    router = router_for({"a": a, "b": b, "c": c}, {"a": ["b"], "b": ["c"]})

    await router.complete(request("a"))

    assert (a.calls, b.calls, c.calls) == (2, 2, 1)


async def test_cycle_does_not_loop_forever() -> None:
    a, b = FakeProvider("a", fail_first=99), FakeProvider("b", fail_first=99)
    router = router_for({"a": a, "b": b}, {"a": ["b"], "b": ["a"]})

    with pytest.raises(ProviderDown):
        await router.complete(request("a"))

    assert (a.calls, b.calls) == (2, 2)


async def test_when_everything_fails_the_last_error_is_raised() -> None:
    primary = FakeProvider("primary", fail_first=99, error_factory=lambda: ProviderDown("p"))
    secondary = FakeProvider("secondary", fail_first=99, error_factory=lambda: Timeout("s"))
    router = router_for({"primary": primary, "secondary": secondary}, {"primary": ["secondary"]})

    with pytest.raises(Timeout):
        await router.complete(request())


async def test_bad_request_does_not_fall_back() -> None:
    primary = FakeProvider("primary", fail_first=99, error_factory=lambda: BadRequest("no"))
    secondary = FakeProvider("secondary")
    router = router_for({"primary": primary, "secondary": secondary}, {"primary": ["secondary"]})

    with pytest.raises(BadRequest):
        await router.complete(request())

    assert secondary.calls == 0


async def test_rate_limited_primary_falls_back() -> None:
    primary = FakeProvider("primary", fail_first=99, error_factory=lambda: RateLimited("slow"))
    secondary = FakeProvider("secondary")
    router = router_for({"primary": primary, "secondary": secondary}, {"primary": ["secondary"]})

    await router.complete(request())

    assert secondary.calls == 1


async def test_fallback_applies_only_to_the_configured_alias() -> None:
    primary = FakeProvider("primary", fail_first=99)
    secondary = FakeProvider("secondary", fail_first=99)
    router = router_for({"primary": primary, "secondary": secondary}, {"primary": ["secondary"]})

    with pytest.raises(ProviderDown):
        await router.complete(request("secondary"))  # no fallback configured for it

    assert primary.calls == 0


async def test_stream_falls_back_before_first_chunk() -> None:
    primary = FakeProvider("primary", fail_first=99)
    secondary = FakeProvider("secondary", reply="x y")
    router = router_for({"primary": primary, "secondary": secondary}, {"primary": ["secondary"]})

    chunks = [c async for c in router.stream(request())]

    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "x y"


class DiesAfterFirstChunk(FakeProvider):
    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        self.calls += 1
        async for chunk in FakeProvider(reply="a b").stream(request):
            yield chunk
            raise ProviderDown("reset", provider="fake")


async def test_stream_does_not_fall_back_once_output_started() -> None:
    primary, secondary = DiesAfterFirstChunk("primary"), FakeProvider("secondary")
    router = router_for({"primary": primary, "secondary": secondary}, {"primary": ["secondary"]})

    with pytest.raises(ProviderDown):
        async for _ in router.stream(request()):
            pass

    assert secondary.calls == 0


def test_stream_with_unknown_alias_still_raises_immediately() -> None:
    router = router_for({"primary": FakeProvider()}, {})

    with pytest.raises(UnknownModelError):
        router.stream(request("nope"))


# ---------- over HTTP ----------


def test_dead_primary_is_served_by_the_fallback_over_http() -> None:
    primary = FakeProvider("primary", fail_first=99)
    secondary = FakeProvider("secondary", reply="backup says hi")

    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return ModelRegistry(
            [deployment("primary", primary), deployment("secondary", secondary)],
            {"primary": ["secondary"]},
        )

    settings = Settings(_env_file=None, retry_base_delay=0.0, retry_max_attempts=2)
    with AuthedClient(create_app(settings, factory)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "primary", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "backup says hi"
