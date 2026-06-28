class ProviderError(Exception):
    """Base for every failure an adapter reports to the router.

    ``retryable`` means the same request may succeed if tried again, on this
    deployment or another one. Non-retryable errors are final.
    """

    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        upstream_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.upstream_status = upstream_status


class RateLimited(ProviderError):
    """The vendor throttled us (HTTP 429). Retry later or elsewhere."""

    retryable = True

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        upstream_status: int | None = 429,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, provider=provider, upstream_status=upstream_status)
        self.retry_after = retry_after


class AuthFailed(ProviderError):
    """Our credentials for this vendor were rejected (401/403). A config problem."""


class Timeout(ProviderError):
    """The vendor did not answer in time (connect or read timeout)."""

    retryable = True


class BadRequest(ProviderError):
    """The vendor rejected the request itself (other 4xx). Retrying cannot help."""


class ProviderDown(ProviderError):
    """The vendor is failing or unreachable (5xx, network error)."""

    retryable = True
