import random
from dataclasses import dataclass
from enum import Enum

from app.providers.errors import (
    AuthFailed,
    BadRequest,
    ProviderError,
    RateLimited,
)


class Decision(Enum):
    RETRY = "retry"  # transient: try again after a backoff, ideally on another deployment
    SWITCH = "switch"  # this deployment is unusable right now: move on without waiting
    FAIL = "fail"  # final: no deployment can do better


def classify(error: BaseException) -> Decision:
    """Decide what the router should do after a failed attempt."""
    if isinstance(error, BadRequest):
        return Decision.FAIL  # the request itself is wrong; every deployment would reject it
    if isinstance(error, RateLimited | AuthFailed):
        # Waiting does not help this deployment (it is throttled or misconfigured), but another
        # deployment, possibly on a different key or vendor, can serve the request now.
        return Decision.SWITCH
    if isinstance(error, ProviderError):
        return Decision.RETRY if error.retryable else Decision.FAIL
    return Decision.FAIL  # a bug on our side is not worth retrying


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3  # total tries, including the first
    base_delay: float = 0.5  # seconds before the first retry
    multiplier: float = 2.0
    max_delay: float = 8.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must not be negative")
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least 1")


def backoff_delay(
    retry_number: int,
    policy: RetryPolicy,
    rng: random.Random,
    retry_after: float | None = None,
) -> float:
    """Seconds to wait before retry number ``retry_number`` (1 = the first retry).

    Exponential growth capped at ``max_delay``, with full jitter (a uniform draw between 0
    and the cap) so many clients retrying together do not stampede. A vendor's ``Retry-After``
    is honoured as a floor, still bounded by ``max_delay``.
    """
    if retry_number < 1:
        raise ValueError("retry_number starts at 1")
    ceiling = min(policy.max_delay, policy.base_delay * policy.multiplier ** (retry_number - 1))
    delay = rng.uniform(0, ceiling)
    if retry_after is not None:
        delay = max(delay, min(retry_after, policy.max_delay))
    return delay
