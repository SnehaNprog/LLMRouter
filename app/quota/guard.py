import logging
import time
from collections.abc import Awaitable, Callable
from typing import Literal

from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

FailureMode = Literal["open", "closed"]
REDIS_FAILURES = (RedisError, OSError, TimeoutError)


class QuotaUnavailableError(Exception):
    """Quotas cannot be checked right now and the gateway is set to refuse rather than guess."""

    def __init__(self, retry_after: float) -> None:
        super().__init__("Quota enforcement is temporarily unavailable. Please retry shortly.")
        self.retry_after = retry_after


class QuotaGuard:
    """What to do when Redis, which holds every quota, cannot be reached.

    ``mode`` is the policy for a request whose limits cannot be checked:

    * ``open``: let it through unenforced. Availability wins; limits are briefly not applied.
    * ``closed``: refuse it with a 503. Limits always win; limited keys are briefly unusable.

    Either way, requests from keys with no limits never reach Redis and are unaffected, and a
    response that has already been produced is never failed because its bookkeeping could not
    be settled.

    A slow or hung Redis would otherwise make *every* request wait out the full timeout before
    the policy applied. So after ``failure_threshold`` failures in a row the guard stops calling
    Redis for ``cooldown`` seconds and applies the policy at once; then it lets one attempt
    through, and a single success closes it again.
    """

    def __init__(
        self,
        mode: FailureMode = "open",
        *,
        failure_threshold: int = 3,
        cooldown: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if mode not in ("open", "closed"):
            raise ValueError("mode must be 'open' or 'closed'")
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if cooldown < 0:
            raise ValueError("cooldown must not be negative")
        self.mode = mode
        self._threshold = failure_threshold
        self._cooldown = cooldown
        self._clock = clock
        self._failures = 0
        self._paused_until = 0.0

    @property
    def paused(self) -> bool:
        """True while Redis is being left alone after repeated failures."""
        return self._cooldown_left() > 0

    def _cooldown_left(self) -> float:
        return max(0.0, self._paused_until - self._clock())

    def _succeeded(self) -> None:
        if self._failures >= self._threshold:
            logger.info("quota backend recovered; enforcing limits again")
        self._failures = 0

    def _failed(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            # Stays at the threshold, so one failed probe after the pause pauses it again.
            self._failures = self._threshold
            was_paused = self.paused
            self._paused_until = self._clock() + self._cooldown
            if not was_paused:
                logger.warning(
                    "quota backend failing; not calling it for %.0fs (fail %s)",
                    self._cooldown,
                    self.mode,
                )

    def _unavailable[T](self, fallback: T) -> T:
        if self.mode == "open":
            return fallback
        raise QuotaUnavailableError(max(1.0, self._cooldown_left()))

    async def enforce[T](
        self, operation: Callable[[], Awaitable[T]], fallback: T, *, what: str
    ) -> T:
        """Run a quota check. If Redis is unavailable: ``fallback`` (open) or raise (closed)."""
        if self.paused:
            return self._unavailable(fallback)
        try:
            result = await operation()
        except REDIS_FAILURES:
            action = "allowing request" if self.mode == "open" else "refusing request"
            logger.warning("%s unavailable, %s", what, action, exc_info=True)
            self._failed()
            return self._unavailable(fallback)
        self._succeeded()
        return result

    async def settle[T](self, operation: Callable[[], Awaitable[T]], *, what: str) -> T | None:
        """Run bookkeeping for a request already served. Never raises: the reservation stands."""
        if self.paused:
            return None
        try:
            result = await operation()
        except REDIS_FAILURES:
            logger.warning("could not settle %s", what, exc_info=True)
            self._failed()
            return None
        self._succeeded()
        return result
