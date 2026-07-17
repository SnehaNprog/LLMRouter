"""Contract every provider adapter must satisfy.

Each adapter gets a Harness that produces canned vendor-format responses; the
tests below are identical for all of them. Adding an adapter means writing one
Harness and passing this suite.
"""

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from app.providers.anthropic import AnthropicProvider
from app.providers.base import Provider
from app.providers.errors import (
    AuthFailed,
    BadRequest,
    ProviderDown,
    ProviderError,
    RateLimited,
    Timeout,
)
from app.providers.openai_compat import OpenAICompatibleProvider
from app.schemas.chat import ChatCompletionRequest
from app.schemas.completion import ChatCompletionChunk

Handler = Callable[[httpx.Request], httpx.Response]

API_KEY = "sk-contract-secret"


def sse(*events: tuple[str | None, dict[str, Any] | str]) -> httpx.Response:
    body = ""
    for name, data in events:
        payload = data if isinstance(data, str) else json.dumps(data)
        body += (f"event: {name}\n" if name else "") + f"data: {payload}\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


@dataclass(frozen=True)
class Harness:
    id: str
    build: Callable[[Handler], Provider]
    text: Callable[[str, int, int], httpx.Response]
    tool: Callable[[str, str, dict[str, Any]], httpx.Response]
    error: Callable[[int, str, dict[str, str] | None], httpx.Response]
    stream_text: Callable[[list[str]], httpx.Response]
    stream_tool: Callable[[str, str, list[str]], httpx.Response]
    stream_error_after_text: Callable[[], httpx.Response]


# ---------------- OpenAI-compatible ----------------


def _oa_chunk(delta: dict[str, Any], finish: str | None = None) -> tuple[None, dict[str, Any]]:
    return None, {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-x",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _oa_completion(message: dict[str, Any], finish: str, usage: tuple[int, int]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-x",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": usage[0],
                "completion_tokens": usage[1],
                "total_tokens": usage[0] + usage[1],
            },
        },
    )


def _oa_usage_chunk(prompt: int, completion: int) -> tuple[None, dict[str, Any]]:
    return None, {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-x",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


def _oa_stream_tool(call_id: str, name: str, fragments: list[str]) -> httpx.Response:
    events: list[tuple[str | None, dict[str, Any] | str]] = [
        _oa_chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }
                ]
            }
        )
    ]
    events += [
        _oa_chunk({"tool_calls": [{"index": 0, "function": {"arguments": f}}]}) for f in fragments
    ]
    events += [_oa_chunk({}, "tool_calls"), _oa_usage_chunk(4, 6), (None, "[DONE]")]
    return sse(*events)


OPENAI = Harness(
    id="openai-compatible",
    build=lambda handler: OpenAICompatibleProvider(
        "openai",
        base_url="https://api.example.com/v1",
        api_key=API_KEY,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ),
    text=lambda text, p, c: _oa_completion({"role": "assistant", "content": text}, "stop", (p, c)),
    tool=lambda call_id, name, args: _oa_completion(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
        },
        "tool_calls",
        (4, 6),
    ),
    error=lambda status, message, headers: httpx.Response(
        status, json={"error": {"message": message}}, headers=headers
    ),
    stream_text=lambda parts: sse(
        _oa_chunk({"role": "assistant", "content": ""}),
        *[_oa_chunk({"content": part}) for part in parts],
        _oa_chunk({}, "stop"),
        _oa_usage_chunk(4, 6),
        (None, "[DONE]"),
    ),
    stream_tool=_oa_stream_tool,
    stream_error_after_text=lambda: sse(
        _oa_chunk({"content": "par"}),
        (None, {"error": {"message": "upstream melted", "type": "api_error"}}),
    ),
)


# ---------------- Anthropic ----------------


def _an_start() -> tuple[str, dict[str, Any]]:
    return "message_start", {
        "type": "message_start",
        "message": {"id": "msg_1", "model": "claude-x", "usage": {"input_tokens": 4}},
    }


def _an_text_delta(text: str) -> tuple[str, dict[str, Any]]:
    return "content_block_delta", {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": text},
    }


def _an_end(reason: str) -> tuple[str, dict[str, Any]]:
    return "message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": reason},
        "usage": {"output_tokens": 6},
    }


def _an_message(
    content: list[dict[str, Any]], stop_reason: str, usage: tuple[int, int]
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-x",
            "content": content,
            "stop_reason": stop_reason,
            "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
        },
    )


def _an_stream_tool(call_id: str, name: str, fragments: list[str]) -> httpx.Response:
    return sse(
        _an_start(),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": call_id, "name": name},
            },
        ),
        *[
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": f},
                },
            )
            for f in fragments
        ],
        _an_end("tool_use"),
        ("message_stop", {"type": "message_stop"}),
    )


ANTHROPIC = Harness(
    id="anthropic",
    build=lambda handler: AnthropicProvider(
        "anthropic",
        api_key=API_KEY,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ),
    text=lambda text, p, c: _an_message([{"type": "text", "text": text}], "end_turn", (p, c)),
    tool=lambda call_id, name, args: _an_message(
        [{"type": "tool_use", "id": call_id, "name": name, "input": args}], "tool_use", (4, 6)
    ),
    error=lambda status, message, headers: httpx.Response(
        status,
        json={"type": "error", "error": {"type": "api_error", "message": message}},
        headers=headers,
    ),
    stream_text=lambda parts: sse(
        _an_start(),
        *[_an_text_delta(part) for part in parts],
        _an_end("end_turn"),
        ("message_stop", {"type": "message_stop"}),
    ),
    stream_tool=_an_stream_tool,
    stream_error_after_text=lambda: sse(
        _an_start(),
        _an_text_delta("par"),
        (
            "error",
            {
                "type": "error",
                "error": {"type": "overloaded_error", "message": "upstream melted"},
            },
        ),
    ),
)


@pytest.fixture(params=[OPENAI, ANTHROPIC], ids=lambda h: h.id)
def harness(request: pytest.FixtureRequest) -> Harness:
    harness_: Harness = request.param
    return harness_


REQUEST = ChatCompletionRequest.model_validate(
    {"model": "gpt-x", "messages": [{"role": "user", "content": "Hi there"}], "stream": True}
)


async def collect(stream: AsyncIterator[ChatCompletionChunk]) -> list[ChatCompletionChunk]:
    return [chunk async for chunk in stream]


def constant(response: httpx.Response) -> Handler:
    return lambda _: response


# ---------------- chat ----------------


def test_provider_has_a_name(harness: Harness) -> None:
    assert harness.build(constant(harness.text("x", 1, 1))).name


async def test_chat_returns_text_and_usage(harness: Harness) -> None:
    provider = harness.build(constant(harness.text("Hello there", 11, 7)))

    response = await provider.chat(REQUEST)

    choice = response.choices[0]
    assert choice.message.role == "assistant"
    assert choice.message.content == "Hello there"
    assert choice.finish_reason == "stop"
    assert response.usage is not None
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (11, 7)
    assert response.usage.total_tokens == 18


async def test_chat_returns_tool_calls(harness: Harness) -> None:
    provider = harness.build(constant(harness.tool("call_1", "get_weather", {"city": "Paris"})))

    response = await provider.chat(REQUEST)

    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls is not None
    call = choice.message.tool_calls[0]
    assert call.id == "call_1"
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"city": "Paris"}


async def test_chat_sends_model_and_prompt_upstream(harness: Harness) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return harness.text("ok", 1, 1)

    await harness.build(handler).chat(REQUEST)

    body = json.loads(seen[0].content)
    assert body["model"] == "gpt-x"
    assert "Hi there" in seen[0].content.decode()
    assert body["stream"] is False


async def test_chat_authenticates_with_the_api_key(harness: Harness) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return harness.text("ok", 1, 1)

    await harness.build(handler).chat(REQUEST)

    assert API_KEY in "".join(seen[0].headers.values())


# ---------------- error mapping ----------------


@pytest.mark.parametrize(
    ("status", "expected", "retryable"),
    [
        (400, BadRequest, False),
        (401, AuthFailed, False),
        (403, AuthFailed, False),
        (404, BadRequest, False),
        (429, RateLimited, True),
        (500, ProviderDown, True),
        (502, ProviderDown, True),
        (503, ProviderDown, True),
    ],
)
async def test_chat_status_maps_to_domain_error(
    harness: Harness, status: int, expected: type[ProviderError], retryable: bool
) -> None:
    provider = harness.build(constant(harness.error(status, "vendor says no", None)))

    with pytest.raises(ProviderError) as excinfo:
        await provider.chat(REQUEST)

    error = excinfo.value
    assert type(error) is expected
    assert error.retryable is retryable
    assert error.upstream_status == status
    assert "vendor says no" in error.message
    assert API_KEY not in str(error)


async def test_chat_rate_limit_carries_retry_after(harness: Harness) -> None:
    provider = harness.build(constant(harness.error(429, "slow", {"Retry-After": "9"})))

    with pytest.raises(RateLimited) as excinfo:
        await provider.chat(REQUEST)

    assert excinfo.value.retry_after == 9.0


async def test_chat_timeout_maps_to_timeout(harness: Harness) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(Timeout):
        await harness.build(handler).chat(REQUEST)


async def test_chat_network_error_maps_to_provider_down(harness: Harness) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderDown):
        await harness.build(handler).chat(REQUEST)


async def test_chat_unreadable_success_body_maps_to_provider_down(harness: Harness) -> None:
    with pytest.raises(ProviderDown):
        await harness.build(constant(httpx.Response(200, text="<html>nope</html>"))).chat(REQUEST)


# ---------------- streaming ----------------


async def test_stream_yields_role_text_and_finish(harness: Harness) -> None:
    provider = harness.build(constant(harness.stream_text(["Hel", "lo ", "there"])))

    chunks = await collect(provider.stream(REQUEST))

    assert chunks[0].choices[0].delta.role == "assistant"
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "Hello there"
    finishes = [
        c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason
    ]
    assert finishes == ["stop"]
    assert len({c.id for c in chunks}) == 1


async def test_stream_reports_usage(harness: Harness) -> None:
    provider = harness.build(constant(harness.stream_text(["a"])))

    chunks = await collect(provider.stream(REQUEST))

    usages = [c.usage for c in chunks if c.usage is not None]
    assert len(usages) == 1
    assert usages[0].total_tokens == usages[0].prompt_tokens + usages[0].completion_tokens == 10


async def test_stream_tool_call_arguments_reassemble(harness: Harness) -> None:
    provider = harness.build(
        constant(harness.stream_tool("call_9", "get_weather", ['{"ci', 'ty":"Rome"}']))
    )

    chunks = await collect(provider.stream(REQUEST))

    calls = [call for c in chunks if c.choices for call in (c.choices[0].delta.tool_calls or [])]
    assert calls[0].id == "call_9"
    assert calls[0].function is not None
    assert calls[0].function.name == "get_weather"
    assert {call.index for call in calls} == {0}
    arguments = "".join(call.function.arguments or "" for call in calls if call.function)
    assert json.loads(arguments) == {"city": "Rome"}
    assert [
        c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason
    ] == ["tool_calls"]


async def test_stream_requests_streaming_upstream(harness: Harness) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return harness.stream_text(["a"])

    await collect(harness.build(handler).stream(REQUEST))

    assert json.loads(seen[0].content)["stream"] is True


@pytest.mark.parametrize(
    ("status", "expected"), [(401, AuthFailed), (429, RateLimited), (503, ProviderDown)]
)
async def test_stream_error_status_raises_before_any_chunk(
    harness: Harness, status: int, expected: type[ProviderError]
) -> None:
    provider = harness.build(constant(harness.error(status, "vendor says no", None)))
    received: list[ChatCompletionChunk] = []

    with pytest.raises(expected):
        async for chunk in provider.stream(REQUEST):
            received.append(chunk)

    assert received == []


async def test_stream_error_after_first_chunk_raises(harness: Harness) -> None:
    provider = harness.build(constant(harness.stream_error_after_text()))
    received: list[ChatCompletionChunk] = []

    with pytest.raises(ProviderDown):
        async for chunk in provider.stream(REQUEST):
            received.append(chunk)

    assert received


async def test_stream_connect_failure_raises_provider_down(harness: Harness) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderDown):
        await collect(harness.build(handler).stream(REQUEST))
