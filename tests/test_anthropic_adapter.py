import json
from typing import Any

import httpx
import pytest

from app.providers.anthropic import AnthropicProvider, translate_request, translate_response
from app.providers.errors import AuthFailed, BadRequest, ProviderDown, RateLimited
from app.schemas.chat import ChatCompletionRequest


def req(**overrides: Any) -> ChatCompletionRequest:
    payload: dict[str, Any] = {
        "model": "claude-haiku-5-5",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    payload.update(overrides)
    return ChatCompletionRequest.model_validate(payload)


# ---------- request translation ----------


def test_basic_request() -> None:
    body = translate_request(req(temperature=0.4, top_p=0.9), default_max_tokens=1000)

    assert body["model"] == "claude-haiku-5-5"
    assert body["max_tokens"] == 1000
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    assert body["temperature"] == 0.4
    assert body["top_p"] == 0.9
    assert body["stream"] is False
    assert "system" not in body


def test_max_tokens_from_request_wins() -> None:
    assert translate_request(req(max_tokens=77), default_max_tokens=1000)["max_tokens"] == 77


def test_system_messages_are_hoisted_and_joined() -> None:
    body = translate_request(
        req(
            messages=[
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hi"},
                {"role": "system", "content": [{"type": "text", "text": "Use French."}]},
            ]
        ),
        1000,
    )

    assert body["system"] == "Be brief.\n\nUse French."
    assert [m["role"] for m in body["messages"]] == ["user"]


def test_temperature_is_clamped_to_one() -> None:
    assert translate_request(req(temperature=1.8), 1000)["temperature"] == 1.0


def test_stop_becomes_stop_sequences() -> None:
    assert translate_request(req(stop="END"), 1000)["stop_sequences"] == ["END"]
    assert translate_request(req(stop=["a", "b"]), 1000)["stop_sequences"] == ["a", "b"]


def test_n_greater_than_one_is_rejected() -> None:
    with pytest.raises(BadRequest):
        translate_request(req(n=2), 1000)


def test_consecutive_same_role_messages_are_merged() -> None:
    body = translate_request(
        req(
            messages=[
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
                {"role": "assistant", "content": "ok"},
            ]
        ),
        1000,
    )

    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assert [b["text"] for b in body["messages"][0]["content"]] == ["one", "two"]


def test_images_url_and_base64() -> None:
    body = translate_request(
        req(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    ],
                }
            ]
        ),
        1000,
    )

    blocks = body["messages"][0]["content"]
    assert blocks[1] == {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}
    assert blocks[2] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
    }


def test_tools_and_tool_choice() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]

    body = translate_request(req(tools=tools, tool_choice="required"), 1000)
    named = translate_request(
        req(tools=tools, tool_choice={"type": "function", "function": {"name": "get_weather"}}),
        1000,
    )

    assert body["tools"] == [
        {
            "name": "get_weather",
            "description": "Weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    assert body["tool_choice"] == {"type": "any"}
    assert named["tool_choice"] == {"type": "tool", "name": "get_weather"}


def test_tool_conversation_round_trip_shape() -> None:
    body = translate_request(
        req(
            messages=[
                {"role": "user", "content": "Weather in Paris and Rome?"},
                {
                    "role": "assistant",
                    "content": "Checking.",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "w", "arguments": '{"city":"Paris"}'}},
                        {"id": "c2", "function": {"name": "w", "arguments": '{"city":"Rome"}'}},
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "18C"},
                {"role": "tool", "tool_call_id": "c2", "content": "24C"},
            ]
        ),
        1000,
    )

    assistant = body["messages"][1]
    assert assistant["content"][0] == {"type": "text", "text": "Checking."}
    assert assistant["content"][1] == {
        "type": "tool_use",
        "id": "c1",
        "name": "w",
        "input": {"city": "Paris"},
    }
    results = body["messages"][2]
    assert results["role"] == "user"
    assert [b["tool_use_id"] for b in results["content"]] == ["c1", "c2"]
    assert results["content"][0]["type"] == "tool_result"


def test_invalid_tool_arguments_are_rejected() -> None:
    with pytest.raises(BadRequest):
        translate_request(
            req(
                messages=[
                    {"role": "user", "content": "x"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "c", "function": {"name": "f", "arguments": "{oops"}}
                        ],
                    },
                ]
            ),
            1000,
        )


# ---------- response translation ----------

ANTHROPIC_REPLY: dict[str, Any] = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "there"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 12, "output_tokens": 4},
}


def test_text_response() -> None:
    response = translate_response(ANTHROPIC_REPLY)

    assert response.id == "msg_01"
    assert response.model == "claude-haiku-5-5"
    assert response.choices[0].message.content == "Hello there"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage is not None
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (12, 4)
    assert response.usage.total_tokens == 16


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "tool_calls"),
        ("refusal", "content_filter"),
        ("something_new", "stop"),
    ],
)
def test_stop_reason_mapping(stop_reason: str, expected: str) -> None:
    response = translate_response({**ANTHROPIC_REPLY, "stop_reason": stop_reason})

    assert response.choices[0].finish_reason == expected


def test_tool_use_response() -> None:
    response = translate_response(
        {
            **ANTHROPIC_REPLY,
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "w", "input": {"city": "Paris"}}
            ],
            "stop_reason": "tool_use",
        }
    )

    message = response.choices[0].message
    assert message.content is None
    assert message.tool_calls is not None
    assert message.tool_calls[0].id == "tu_1"
    assert message.tool_calls[0].function.name == "w"
    assert json.loads(message.tool_calls[0].function.arguments) == {"city": "Paris"}
    assert response.choices[0].finish_reason == "tool_calls"


# ---------- adapter over HTTP ----------


def provider(handler: Any) -> AnthropicProvider:
    return AnthropicProvider(
        api_key="sk-ant-test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def test_chat_sends_anthropic_headers_and_url() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ANTHROPIC_REPLY)

    response = await provider(handler).chat(req())

    sent = seen[0]
    assert str(sent.url) == "https://api.anthropic.com/v1/messages"
    assert sent.headers["x-api-key"] == "sk-ant-test"
    assert sent.headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in sent.headers
    assert json.loads(sent.content)["max_tokens"] == 4096
    assert response.choices[0].message.content == "Hello there"


@pytest.mark.parametrize(
    ("status", "expected"), [(401, AuthFailed), (429, RateLimited), (529, ProviderDown)]
)
async def test_errors_use_shared_mapping(status: int, expected: type[Exception]) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, json={"type": "error", "error": {"type": "x", "message": "nope"}}
        )

    with pytest.raises(expected) as excinfo:
        await provider(handler).chat(req())

    assert "nope" in str(excinfo.value)


async def test_garbage_success_body_becomes_provider_down() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    with pytest.raises(ProviderDown):
        await provider(handler).chat(req())
