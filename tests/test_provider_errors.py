import pytest

from app.providers.errors import (
    AuthFailed,
    BadRequest,
    ProviderDown,
    ProviderError,
    RateLimited,
    Timeout,
)


@pytest.mark.parametrize(
    ("error_type", "retryable"),
    [
        (RateLimited, True),
        (Timeout, True),
        (ProviderDown, True),
        (AuthFailed, False),
        (BadRequest, False),
        (ProviderError, False),
    ],
)
def test_retryable_flag(error_type: type[ProviderError], retryable: bool) -> None:
    assert error_type("x").retryable is retryable


def test_all_errors_share_a_base() -> None:
    for error_type in (RateLimited, AuthFailed, Timeout, BadRequest, ProviderDown):
        assert issubclass(error_type, ProviderError)


def test_context_is_kept() -> None:
    error = ProviderDown("boom", provider="openai", upstream_status=503)

    assert str(error) == "boom"
    assert error.provider == "openai"
    assert error.upstream_status == 503


def test_rate_limited_carries_retry_after() -> None:
    error = RateLimited("slow down", provider="groq", retry_after=7.5)

    assert error.retry_after == 7.5
    assert error.upstream_status == 429


def test_rate_limited_retry_after_is_optional() -> None:
    assert RateLimited("x").retry_after is None
