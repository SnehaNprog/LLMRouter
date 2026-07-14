import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from app.providers.anthropic import AnthropicProvider
from app.providers.errors import AuthFailed, ProviderDown, RateLimited, Timeout
from app.providers.openai_compat import OpenAICompatibleProvider
from app.providers.sse import SSEEvent, iter_sse
from app.schemas.chat import ChatCompletionRequest, Message
from app.schemas.completion import ChatCompletionChunk

REQUEST = ChatCompletionRequest(
    model="m", messages=[Message(role="user", content="hi")], stream=True
)


def sse_response(body: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status, content=body.encode(), headers={"content-type": "text/event-stream"}
    )


def client_for(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def collect(stream: AsyncIterator[ChatCompletionChunk]) -> list[ChatCompletionChunk]:
    return [chunk async for chunk in stream]


# ---------- SSE parser ----------


async def parse(body: str) -> list[SSEEvent]:
    return [event async for event in iter_sse(sse_response(body))]


async def test_sse_parses_data_and_named_events() -> None:
    events = await parse("event: a\ndata: 1\n\ndata: 2\n\n")

    assert events == [SSEEvent("a", "1"), SSEEvent(None, "2")]


async def test_sse_skips_comments_and_joins_multiline_data() -> None:
    events = await parse(": keep-alive\ndata: line1\ndata: line2\n\n")

    assert events == [SSEEvent(None, "line1\nline2")]


async def test_sse_flushes_last_event_without_trailing_blank_line() -> None:
    assert await parse("data: tail") == [SSEEvent(None, "tail")]


# ---------- OpenAI-compatible ----------


def openai_chunk(delta: dict[str, Any], finish: str | None = None) -> str:
    body = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(body)}\n\n"


def openai_provider(handler: Any) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        "groq", base_url="https://api.example.com/v1", api_key="sk", client=client_for(handler)
    )


async def test_openai_stream_yields_chunks_until_done() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return sse_response(
            openai_chunk({"role": "assistant", "content": ""})
            + openai_chunk({"content": "Hel"})
            + openai_chunk({"content": "lo"})
            + openai_chunk({}, "stop")
            + 'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m",'
            '"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
            + "data: [DONE]\n\n"
        )

    chunks = await collect(openai_provider(handler).stream(REQUEST))

    body = json.loads(seen[0].content)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == "Hello"
    assert chunks[-2].choices[0].finish_reason == "stop"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 5


async def test_openai_stream_stops_at_done_marker() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return sse_response(
            openai_chunk({"content": "a"})
            + "data: [DONE]\n\n"
            + openai_chunk({"content": "ignored"})
        )

    chunks = await collect(openai_provider(handler).stream(REQUEST))

    assert len(chunks) == 1


@pytest.mark.parametrize(
    ("status", "expected"), [(429, RateLimited), (401, AuthFailed), (503, ProviderDown)]
)
async def test_openai_error_status_raises_before_any_chunk(
    status: int, expected: type[Exception]
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "nope"}})

    received: list[ChatCompletionChunk] = []
    with pytest.raises(expected):
        async for chunk in openai_provider(handler).stream(REQUEST):
            received.append(chunk)

    assert received == []


async def test_openai_error_object_mid_stream_raises() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return sse_response(
            openai_chunk({"content": "a"})
            + 'data: {"error":{"message":"overloaded","type":"api_error"}}\n\n'
        )

    received: list[ChatCompletionChunk] = []
    with pytest.raises(ProviderDown):
        async for chunk in openai_provider(handler).stream(REQUEST):
            received.append(chunk)

    assert len(received) == 1


async def test_openai_garbage_chunk_becomes_provider_down() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return sse_response("data: not json\n\n")

    with pytest.raises(ProviderDown):
        await collect(openai_provider(handler).stream(REQUEST))


class BrokenStream(httpx.AsyncByteStream):
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield openai_chunk({"content": "partial"}).encode()
        raise self._exc


async def test_connection_drop_mid_stream_becomes_provider_down() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=BrokenStream(httpx.ReadError("reset")))

    received: list[ChatCompletionChunk] = []
    with pytest.raises(ProviderDown):
        async for chunk in openai_provider(handler).stream(REQUEST):
            received.append(chunk)

    assert len(received) == 1


async def test_read_timeout_mid_stream_becomes_timeout() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=BrokenStream(httpx.ReadTimeout("slow")))

    with pytest.raises(Timeout):
        await collect(openai_provider(handler).stream(REQUEST))


async def test_connect_failure_before_stream_raises_provider_down() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderDown):
        await collect(openai_provider(handler).stream(REQUEST))


# ---------- Anthropic ----------


def anthropic_event(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps({'type': name, **data})}\n\n"


def anthropic_provider(handler: Any) -> AnthropicProvider:
    return AnthropicProvider(api_key="sk-ant", client=client_for(handler))


TEXT_STREAM = (
    anthropic_event(
        "message_start",
        {"message": {"id": "msg_1", "model": "claude-x", "usage": {"input_tokens": 12}}},
    )
    + anthropic_event(
        "content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}
    )
    + 'event: ping\ndata: {"type": "ping"}\n\n'
    + anthropic_event(
        "content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "Hel"}}
    )
    + anthropic_event(
        "content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "lo"}}
    )
    + anthropic_event("content_block_stop", {"index": 0})
    + anthropic_event(
        "message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}}
    )
    + anthropic_event("message_stop", {})
)


async def test_anthropic_text_stream_is_translated() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return sse_response(TEXT_STREAM)

    chunks = await collect(anthropic_provider(handler).stream(REQUEST))

    assert json.loads(seen[0].content)["stream"] is True
    assert chunks[0].choices[0].delta.role == "assistant"
    assert all(c.id == "msg_1" and c.model == "claude-x" for c in chunks)
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "Hello"
    last = chunks[-1]
    assert last.choices[0].finish_reason == "stop"
    assert last.usage is not None
    assert (last.usage.prompt_tokens, last.usage.completion_tokens) == (12, 4)


async def test_anthropic_tool_use_stream_is_translated() -> None:
    body = (
        anthropic_event(
            "message_start",
            {"message": {"id": "msg_2", "model": "claude-x", "usage": {"input_tokens": 5}}},
        )
        + anthropic_event(
            "content_block_start",
            {"index": 0, "content_block": {"type": "text", "text": ""}},
        )
        + anthropic_event(
            "content_block_start",
            {"index": 1, "content_block": {"type": "tool_use", "id": "tu_1", "name": "w"}},
        )
        + anthropic_event(
            "content_block_delta",
            {"index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"ci'}},
        )
        + anthropic_event(
            "content_block_delta",
            {"index": 1, "delta": {"type": "input_json_delta", "partial_json": 'ty":"Paris"}'}},
        )
        + anthropic_event(
            "message_delta",
            {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
        )
    )

    chunks = await collect(anthropic_provider(lambda _: sse_response(body)).stream(REQUEST))

    calls = [call for c in chunks for call in (c.choices[0].delta.tool_calls or [])]
    assert calls[0].id == "tu_1"
    assert calls[0].index == 0
    assert calls[0].function is not None
    assert calls[0].function.name == "w"
    arguments = "".join(c.function.arguments or "" for c in calls if c.function)
    assert json.loads(arguments) == {"city": "Paris"}
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


@pytest.mark.parametrize(
    ("status", "expected"), [(429, RateLimited), (401, AuthFailed), (529, ProviderDown)]
)
async def test_anthropic_error_status_raises_before_any_chunk(
    status: int, expected: type[Exception]
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, json={"type": "error", "error": {"type": "x", "message": "nope"}}
        )

    received: list[ChatCompletionChunk] = []
    with pytest.raises(expected):
        async for chunk in anthropic_provider(handler).stream(REQUEST):
            received.append(chunk)

    assert received == []


async def test_anthropic_error_event_mid_stream_raises() -> None:
    body = anthropic_event(
        "message_start",
        {"message": {"id": "m", "model": "x", "usage": {"input_tokens": 1}}},
    ) + anthropic_event("error", {"error": {"type": "overloaded_error", "message": "Overloaded"}})

    received: list[ChatCompletionChunk] = []
    with pytest.raises(ProviderDown):
        async for chunk in anthropic_provider(lambda _: sse_response(body)).stream(REQUEST):
            received.append(chunk)

    assert len(received) == 1


async def test_anthropic_rate_limit_error_event_mid_stream() -> None:
    body = anthropic_event("error", {"error": {"type": "rate_limit_error", "message": "slow"}})

    with pytest.raises(RateLimited):
        await collect(anthropic_provider(lambda _: sse_response(body)).stream(REQUEST))


async def test_anthropic_garbage_event_becomes_provider_down() -> None:
    with pytest.raises(ProviderDown):
        await collect(
            anthropic_provider(
                lambda _: sse_response("event: message_start\ndata: {oops\n\n")
            ).stream(REQUEST)
        )
