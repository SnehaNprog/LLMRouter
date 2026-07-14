from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from app.providers.errors import (
    AuthFailed,
    BadRequest,
    ProviderDown,
    ProviderError,
    RateLimited,
    Timeout,
)


def _upstream_message(response: httpx.Response) -> str:
    try:
        body = response.json()
        message = body["error"]["message"]
        if isinstance(message, str) and message:
            return message
    except (ValueError, KeyError, TypeError):
        pass
    return f"HTTP {response.status_code}"


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # HTTP-date form is not supported


def error_for_response(provider: str, response: httpx.Response) -> ProviderError:
    """Translate a non-2xx vendor response into a domain error."""
    status = response.status_code
    message = f"{provider}: {_upstream_message(response)}"
    if status in (401, 403):
        return AuthFailed(message, provider=provider, upstream_status=status)
    if status == 429:
        return RateLimited(
            message, provider=provider, upstream_status=status, retry_after=_retry_after(response)
        )
    if status == 408:
        return Timeout(message, provider=provider, upstream_status=status)
    if status >= 500:
        return ProviderDown(message, provider=provider, upstream_status=status)
    return BadRequest(message, provider=provider, upstream_status=status)


@asynccontextmanager
async def translate_transport_errors(provider: str) -> AsyncIterator[None]:
    """Turn httpx timeouts and network failures into domain errors."""
    try:
        yield
    except httpx.TimeoutException as exc:
        raise Timeout(f"{provider}: request timed out", provider=provider) from exc
    except httpx.TransportError as exc:
        raise ProviderDown(
            f"{provider}: network error ({type(exc).__name__})", provider=provider
        ) from exc


def error_for_stream_event(provider: str, error_type: str, message: str) -> ProviderError:
    """Translate an error event received in the middle of a stream."""
    text = f"{provider}: {message}"
    if error_type == "rate_limit_error":
        return RateLimited(text, provider=provider)
    if error_type == "authentication_error" or error_type == "permission_error":
        return AuthFailed(text, provider=provider)
    if error_type in ("invalid_request_error", "not_found_error"):
        return BadRequest(text, provider=provider)
    return ProviderDown(text, provider=provider)  # overloaded_error, api_error, unknown
