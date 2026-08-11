import logging
from typing import Any

import pytest

from app.core.model_config import ConfigError, parse_config
from app.providers.fake import FakeProvider
from app.router.cost import cost_usd
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry
from app.router.retry import RetryPolicy
from app.router.router import Router
from app.router.tokens import estimate_prompt_tokens, estimate_text_tokens
from app.schemas.chat import ChatCompletionRequest
from tests.conftest import TEST_ALIAS, model_config


def request(messages: list[dict[str, Any]], **extra: Any) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate({"model": "m", "messages": messages, **extra})


def user(content: Any) -> dict[str, Any]:
    return {"role": "user", "content": content}


# ---------- text estimates ----------


def test_empty_text_is_zero_tokens() -> None:
    assert estimate_text_tokens("") == 0


@pytest.mark.parametrize(("text", "expected"), [("abcd", 1), ("abcde", 2), ("a" * 400, 100)])
def test_ascii_is_about_four_characters_per_token(text: str, expected: int) -> None:
    assert estimate_text_tokens(text) == expected


def test_non_ascii_costs_roughly_a_token_per_character() -> None:
    assert estimate_text_tokens("你好世界") == 4
    assert estimate_text_tokens("日本語のテキスト") == 8


def test_mixed_text_adds_both_parts() -> None:
    assert estimate_text_tokens("abcd你好") == 1 + 2


def test_realistic_sentence_is_in_the_right_ballpark() -> None:
    tokens = estimate_text_tokens("Hello, how are you doing today? I hope you are well.")

    assert 8 <= tokens <= 18  # a real tokenizer gives about 13


def test_longer_text_never_estimates_fewer_tokens() -> None:
    text = "The quick brown fox jumps over the lazy dog. "
    counts = [estimate_text_tokens(text * n) for n in range(1, 20)]

    assert counts == sorted(counts)


# ---------- prompt estimates ----------


def test_single_message_includes_framing() -> None:
    tokens = estimate_prompt_tokens(request([user("abcd")]))

    assert tokens == 3 + 4 + 1  # reply priming + message framing + one token of text


def test_messages_add_up() -> None:
    one = estimate_prompt_tokens(request([user("hello there")]))
    two = estimate_prompt_tokens(request([user("hello there"), user("hello there")]))

    assert two - 3 == 2 * (one - 3)


def test_text_parts_are_counted() -> None:
    plain = estimate_prompt_tokens(request([user("a" * 40)]))
    parts = estimate_prompt_tokens(request([user([{"type": "text", "text": "a" * 40}])]))

    assert plain == parts


def test_images_cost_more_than_nothing_and_low_detail_costs_less() -> None:
    def with_image(detail: str | None) -> int:
        image: dict[str, Any] = {"url": "https://x/y.png"}
        if detail:
            image["detail"] = detail
        return estimate_prompt_tokens(request([user([{"type": "image_url", "image_url": image}])]))

    assert with_image("low") == 3 + 4 + 85
    assert with_image(None) == 3 + 4 + 765
    assert with_image("high") == with_image(None)


def test_tool_definitions_are_counted() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    without = estimate_prompt_tokens(request([user("hi")]))
    with_tools = estimate_prompt_tokens(request([user("hi")], tools=tools))

    assert with_tools > without + 15


def test_assistant_tool_calls_are_counted() -> None:
    plain = request([user("hi"), {"role": "assistant", "content": "ok"}])
    calling = request(
        [
            user("hi"),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
        ]
    )

    assert estimate_prompt_tokens(calling) > estimate_prompt_tokens(plain)


def test_message_name_is_counted() -> None:
    plain = estimate_prompt_tokens(request([user("hi")]))
    named = estimate_prompt_tokens(request([{**user("hi"), "name": "alice"}]))

    assert named > plain


# ---------- cost ----------


def test_cost_from_per_million_rates() -> None:
    config = model_config(input_cost_per_mtok=3.0, output_cost_per_mtok=15.0)

    assert cost_usd(config, 1_000_000, 0) == pytest.approx(3.0)
    assert cost_usd(config, 0, 1_000_000) == pytest.approx(15.0)
    assert cost_usd(config, 2_000, 500) == pytest.approx(0.006 + 0.0075)


def test_free_models_cost_nothing() -> None:
    assert cost_usd(model_config(), 123_456, 7_890) == 0.0


def test_zero_tokens_cost_nothing() -> None:
    assert cost_usd(model_config(input_cost_per_mtok=3.0, output_cost_per_mtok=15.0), 0, 0) == 0.0


def test_negative_tokens_are_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        cost_usd(model_config(), -1, 0)


def test_deployment_exposes_cost() -> None:
    deployment = Deployment.create(
        model_config(input_cost_per_mtok=2.0, output_cost_per_mtok=4.0), FakeProvider()
    )

    assert deployment.cost(1_000_000, 500_000) == pytest.approx(4.0)


# ---------- cost in config ----------


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


def test_rates_are_read_from_config() -> None:
    config = parse_config(entry(input_cost_per_mtok=0.5, output_cost_per_mtok=1.5), {}).models[0]

    assert (config.input_cost_per_mtok, config.output_cost_per_mtok) == (0.5, 1.5)


def test_rates_default_to_free() -> None:
    config = parse_config(entry(), {}).models[0]

    assert (config.input_cost_per_mtok, config.output_cost_per_mtok) == (0.0, 0.0)


@pytest.mark.parametrize("field", ["input_cost_per_mtok", "output_cost_per_mtok"])
def test_negative_rates_are_rejected(field: str) -> None:
    with pytest.raises(ConfigError, match=field):
        parse_config(entry(**{field: -1}), {})


# ---------- cost is computed per request ----------


async def test_router_logs_the_cost_of_each_completion(caplog: pytest.LogCaptureFixture) -> None:
    provider = FakeProvider(prompt_tokens=2_000, completion_tokens=500)
    registry = ModelRegistry(
        [
            Deployment.create(
                model_config(input_cost_per_mtok=3.0, output_cost_per_mtok=15.0), provider
            )
        ]
    )

    with caplog.at_level(logging.INFO, logger="app.router.router"):
        await Router(registry).complete(request([user("hi")], model=TEST_ALIAS))

    record = next(r for r in caplog.records if r.getMessage() == "completion")
    assert record.prompt_tokens == 2_000  # type: ignore[attr-defined]
    assert record.completion_tokens == 500  # type: ignore[attr-defined]
    assert record.cost_usd == pytest.approx(0.0135)  # type: ignore[attr-defined]


async def test_cost_is_logged_for_the_deployment_that_actually_answered(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cheap = Deployment.create(
        model_config(name="cheap", model="cheap-model", input_cost_per_mtok=1.0),
        FakeProvider(fail_first=99),
    )
    pricey = Deployment.create(
        model_config(name="pricey", model="pricey-model", input_cost_per_mtok=100.0),
        FakeProvider(prompt_tokens=1_000_000, completion_tokens=0),
    )
    router = Router(
        ModelRegistry([cheap, pricey], {"cheap": ["pricey"]}), policy=RetryPolicy(max_attempts=1)
    )

    with caplog.at_level(logging.INFO, logger="app.router.router"):
        await router.complete(request([user("hi")], model="cheap"))

    record = next(r for r in caplog.records if r.getMessage() == "completion")
    assert record.model == "pricey-model"  # type: ignore[attr-defined]
    assert record.cost_usd == pytest.approx(100.0)  # type: ignore[attr-defined]


async def test_streaming_logs_cost_when_usage_arrives(caplog: pytest.LogCaptureFixture) -> None:
    class StreamsUsage(FakeProvider):
        async def stream(self, request: ChatCompletionRequest):  # type: ignore[no-untyped-def]
            async for chunk in super().stream(request):
                yield chunk
            from app.schemas.completion import ChatCompletionChunk, Usage

            yield ChatCompletionChunk(
                id="x", created=0, model="m", choices=[], usage=Usage.of(1_000_000, 0)
            )

    registry = ModelRegistry(
        [Deployment.create(model_config(input_cost_per_mtok=2.0), StreamsUsage())]
    )

    with caplog.at_level(logging.INFO, logger="app.router.router"):
        async for _ in Router(registry).stream(request([user("hi")], model=TEST_ALIAS)):
            pass

    records = [r for r in caplog.records if r.getMessage() == "completion"]
    assert len(records) == 1
    assert records[0].cost_usd == pytest.approx(2.0)  # type: ignore[attr-defined]


def test_shipped_example_has_prices() -> None:
    from pathlib import Path

    from app.core.model_config import load_config

    config = load_config(
        Path(__file__).parent.parent / "config.example.yaml",
        {"GROQ_API_KEY": "a", "ANTHROPIC_API_KEY": "b"},
    )

    prices = {m.name: m.input_cost_per_mtok for m in config.models}
    assert prices["smart"] > prices["fast"] > prices["local"] == 0.0
