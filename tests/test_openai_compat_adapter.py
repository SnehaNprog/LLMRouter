import json

import httpx
import pytest

from app.providers.errors import ProviderError
from app.providers.openai_compat import OpenAICompatibleProvider
from app.schemas.chat import ChatCompletionRequest, Message

UPSTREAM_REPLY = {
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "created": 1730000000,
    "model": "llama-3.1-8b-instant",
    "system_fingerprint": "fp_123",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello there"},
            "finish_reason": "stop",
            "logprobs": None,
        }
    ],
    "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
}


def make_provider(
    captured: list[httpx.Request] | None = None,
    reply: dict[str, object] | None = None,
    status: int = 200,
    base_url: str = "https://api.example.com/v1",
    api_key: str | None = "sk-test",
) -> OpenAICompatibleProvider:
    def respond(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        return httpx.Response(status, json=reply if reply is not None else UPSTREAM_REPLY)

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    return OpenAICompatibleProvider("groq", base_url=base_url, api_key=api_key, client=client)


REQUEST = ChatCompletionRequest(
    model="llama-3.1-8b-instant",
    messages=[Message(role="user", content="Hi")],
    temperature=0.3,
    stream=True,
)


async def test_returns_parsed_completion() -> None:
    response = await make_provider().chat(REQUEST)

    assert response.choices[0].message.content == "Hello there"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.total_tokens == 12


async def test_sends_openai_shaped_request() -> None:
    captured: list[httpx.Request] = []

    await make_provider(captured=captured).chat(REQUEST)

    sent = captured[0]
    body = json.loads(sent.content)
    assert sent.method == "POST"
    assert str(sent.url) == "https://api.example.com/v1/chat/completions"
    assert sent.headers["Authorization"] == "Bearer sk-test"
    assert body["model"] == "llama-3.1-8b-instant"
    assert body["messages"] == [{"role": "user", "content": "Hi"}]
    assert body["temperature"] == 0.3
    assert body["stream"] is False
    assert "max_tokens" not in body


async def test_trailing_slash_in_base_url_is_handled() -> None:
    captured: list[httpx.Request] = []

    await make_provider(captured=captured, base_url="https://api.example.com/v1/").chat(REQUEST)

    assert str(captured[0].url) == "https://api.example.com/v1/chat/completions"


async def test_no_authorization_header_without_key() -> None:
    captured: list[httpx.Request] = []

    await make_provider(captured=captured, api_key=None).chat(REQUEST)

    assert "Authorization" not in captured[0].headers


async def test_tool_call_reply_is_parsed() -> None:
    reply = {
        **UPSTREAM_REPLY,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    response = await make_provider(reply=reply).chat(REQUEST)

    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls is not None


async def test_http_error_becomes_provider_error() -> None:
    provider = make_provider(status=500, reply={"error": {"message": "boom"}})

    with pytest.raises(ProviderError) as excinfo:
        await provider.chat(REQUEST)

    assert excinfo.value.upstream_status == 500
    assert excinfo.value.provider == "groq"
