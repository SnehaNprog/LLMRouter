import httpx
import pytest

from app.core.config import Settings
from app.core.http import build_http_client
from app.providers.errors import (
    AuthFailed,
    BadRequest,
    ProviderDown,
    ProviderError,
    RateLimited,
    Timeout,
)
from app.providers.openai_compat import OpenAICompatibleProvider
from app.schemas.chat import ChatCompletionRequest, Message

REQUEST = ChatCompletionRequest(model="m", messages=[Message(role="user", content="hi")])


def provider_with(handler: httpx.MockTransport) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        "groq",
        base_url="https://api.example.com/v1",
        api_key="sk-secret",
        client=httpx.AsyncClient(transport=handler),
    )


def status_provider(status: int, headers: dict[str, str] | None = None) -> OpenAICompatibleProvider:
    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, json={"error": {"message": "upstream says no"}}, headers=headers
        )

    return provider_with(httpx.MockTransport(respond))


@pytest.mark.parametrize(
    ("status", "expected", "retryable"),
    [
        (400, BadRequest, False),
        (401, AuthFailed, False),
        (403, AuthFailed, False),
        (404, BadRequest, False),
        (408, Timeout, True),
        (422, BadRequest, False),
        (429, RateLimited, True),
        (500, ProviderDown, True),
        (502, ProviderDown, True),
        (503, ProviderDown, True),
    ],
)
async def test_status_maps_to_domain_error(
    status: int, expected: type[ProviderError], retryable: bool
) -> None:
    with pytest.raises(ProviderError) as excinfo:
        await status_provider(status).chat(REQUEST)

    error = excinfo.value
    assert type(error) is expected
    assert error.retryable is retryable
    assert error.upstream_status == status
    assert error.provider == "groq"
    assert "upstream says no" in error.message


async def test_retry_after_seconds_is_parsed() -> None:
    with pytest.raises(RateLimited) as excinfo:
        await status_provider(429, {"Retry-After": "12"}).chat(REQUEST)

    assert excinfo.value.retry_after == 12.0


async def test_retry_after_http_date_is_ignored() -> None:
    with pytest.raises(RateLimited) as excinfo:
        await status_provider(429, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}).chat(REQUEST)

    assert excinfo.value.retry_after is None


async def test_error_with_non_json_body_still_maps() -> None:
    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html>Bad gateway</html>")

    with pytest.raises(ProviderDown) as excinfo:
        await provider_with(httpx.MockTransport(respond)).chat(REQUEST)

    assert "HTTP 502" in excinfo.value.message


async def test_slow_server_becomes_timeout() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(Timeout) as excinfo:
        await provider_with(httpx.MockTransport(respond)).chat(REQUEST)

    assert excinfo.value.retryable is True


async def test_connect_timeout_becomes_timeout() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("no route", request=request)

    with pytest.raises(Timeout):
        await provider_with(httpx.MockTransport(respond)).chat(REQUEST)


async def test_network_error_becomes_provider_down() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderDown) as excinfo:
        await provider_with(httpx.MockTransport(respond)).chat(REQUEST)

    assert excinfo.value.retryable is True


async def test_garbage_success_body_becomes_provider_down() -> None:
    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json at all")

    with pytest.raises(ProviderDown):
        await provider_with(httpx.MockTransport(respond)).chat(REQUEST)


async def test_wrong_shape_success_body_becomes_provider_down() -> None:
    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    with pytest.raises(ProviderDown):
        await provider_with(httpx.MockTransport(respond)).chat(REQUEST)


async def test_api_key_never_appears_in_error() -> None:
    with pytest.raises(ProviderError) as excinfo:
        await status_provider(401).chat(REQUEST)

    assert "sk-secret" not in str(excinfo.value)


def test_client_uses_timeouts_from_settings() -> None:
    settings = Settings(
        _env_file=None,
        http_connect_timeout=1.5,
        http_read_timeout=20.0,
        http_write_timeout=2.5,
        http_pool_timeout=0.5,
    )

    timeout = build_http_client(settings).timeout

    assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (1.5, 20.0, 2.5, 0.5)
