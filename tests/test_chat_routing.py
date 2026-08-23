import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.providers.errors import (
    AuthFailed,
    BadRequest,
    ProviderDown,
    ProviderError,
    RateLimited,
    Timeout,
)
from app.providers.fake import FakeProvider
from app.router.registry import ModelRegistry
from app.schemas.chat import ChatCompletionRequest
from app.schemas.completion import ChatCompletionChunk
from tests.conftest import UPSTREAM_MODEL, AuthedClient, registry_with

BODY = {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]}


def client_for(provider: FakeProvider, attempts: int = 1) -> AuthedClient:
    """Single attempt by default, so error-mapping tests do not exercise (or wait on) retries."""

    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return registry_with(provider)

    settings = Settings(_env_file=None, retry_base_delay=0.0, retry_max_attempts=attempts)
    return AuthedClient(create_app(settings, factory))


def sse_events(text: str) -> list[str]:
    return [block.removeprefix("data: ") for block in text.split("\n\n") if block]


# ---------- non-streaming ----------


def test_request_is_served_by_the_provider(client: TestClient, provider: FakeProvider) -> None:
    response = client.post("/v1/chat/completions", json=BODY)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == provider.reply
    assert provider.requests[0].model == UPSTREAM_MODEL


def test_unknown_model_is_an_openai_shaped_404(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json={**BODY, "model": "gpt-9"})

    assert response.status_code == 404
    assert response.json()["error"] == {
        "message": "The model 'gpt-9' does not exist",
        "type": "invalid_request_error",
        "param": "model",
        "code": "model_not_found",
    }


@pytest.mark.parametrize(
    ("error", "status", "error_type", "code"),
    [
        (
            RateLimited("slow down", retry_after=2.2),
            429,
            "rate_limit_error",
            "upstream_rate_limited",
        ),
        (AuthFailed("bad key sk-secret"), 502, "api_error", "upstream_auth_failed"),
        (Timeout("too slow"), 504, "api_error", "upstream_timeout"),
        (BadRequest("context too long"), 400, "invalid_request_error", "upstream_bad_request"),
        (ProviderDown("api.example.com: overloaded"), 502, "api_error", "upstream_unavailable"),
    ],
)
def test_provider_errors_map_to_gateway_errors(
    error: ProviderError, status: int, error_type: str, code: str
) -> None:
    client = client_for(FakeProvider(fail_first=99, error_factory=lambda: error))

    with client:
        response = client.post("/v1/chat/completions", json=BODY)

    assert response.status_code == status
    body = response.json()["error"]
    assert body["type"] == error_type
    assert body["code"] == code
    assert "sk-secret" not in response.text


def test_rate_limit_keeps_retry_after() -> None:
    client = client_for(
        FakeProvider(fail_first=99, error_factory=lambda: RateLimited("x", retry_after=2.2))
    )

    with client:
        response = client.post("/v1/chat/completions", json=BODY)

    assert response.headers["Retry-After"] == "3"


# ---------- streaming ----------


def test_stream_is_served_as_server_sent_events(provider: FakeProvider) -> None:
    provider.reply = "one two"
    with client_for(provider) as client:
        response = client.post("/v1/chat/completions", json={**BODY, "stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response.text)
    assert events[-1] == "[DONE]"
    chunks = [ChatCompletionChunk.model_validate_json(e) for e in events[:-1]]
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "one two"
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert provider.requests[0].model == UPSTREAM_MODEL


def test_stream_unknown_model_is_a_normal_404(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json={**BODY, "model": "nope", "stream": True})

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


def test_stream_failure_before_first_chunk_is_a_real_http_error() -> None:
    client = client_for(
        FakeProvider(fail_first=99, error_factory=lambda: RateLimited("slow", retry_after=1))
    )

    with client:
        response = client.post("/v1/chat/completions", json={**BODY, "stream": True})

    assert response.status_code == 429
    assert response.headers["content-type"].startswith("application/json")


class DiesMidStream(FakeProvider):
    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        sent = 0
        async for chunk in super().stream(request):
            yield chunk
            sent += 1
            if sent == 2:
                raise ProviderDown("api.example.com: connection reset")


def test_failure_mid_stream_is_reported_as_an_error_event() -> None:
    with client_for(DiesMidStream(reply="a b c d")) as client:
        response = client.post("/v1/chat/completions", json={**BODY, "stream": True})

    assert response.status_code == 200  # headers were already sent
    events = sse_events(response.text)
    assert "[DONE]" not in events
    error: dict[str, Any] = json.loads(events[-1])
    assert error["error"]["code"] == "upstream_unavailable"
    assert len(events) == 3  # two chunks, then the error


def test_openai_sdk_can_stream(provider: FakeProvider, app_: FastAPI) -> None:
    from openai import OpenAI

    provider.reply = "streamed words here"
    with AuthedClient(app_) as http:
        sdk = OpenAI(base_url="http://testserver/v1", api_key=http.secret, http_client=http)
        stream = sdk.chat.completions.create(
            model="test-model", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)

    assert text == "streamed words here"


def test_transient_failures_are_retried_transparently() -> None:
    provider = FakeProvider(fail_first=2)

    with client_for(provider, attempts=3) as client:
        response = client.post("/v1/chat/completions", json=BODY)

    assert response.status_code == 200
    assert provider.calls == 3


def test_stream_start_failures_are_retried_transparently() -> None:
    provider = FakeProvider(reply="ok then", fail_first=2)

    with client_for(provider, attempts=3) as client:
        response = client.post("/v1/chat/completions", json={**BODY, "stream": True})

    assert response.status_code == 200
    assert sse_events(response.text)[-1] == "[DONE]"
    assert provider.calls == 3


def test_gives_up_after_max_attempts() -> None:
    provider = FakeProvider(fail_first=99)

    with client_for(provider, attempts=3) as client:
        response = client.post("/v1/chat/completions", json=BODY)

    assert response.status_code == 502
    assert provider.calls == 3
