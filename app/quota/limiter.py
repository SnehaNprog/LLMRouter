import logging
from collections.abc import Callable

from redis.exceptions import RedisError

from app.auth.dependencies import AuthContext
from app.quota.bucket import BucketStore

logger = logging.getLogger(__name__)

_SECONDS_PER_MINUTE = 60


class RateLimitExceededError(Exception):
    def __init__(self, limit: int, retry_after: float | None) -> None:
        wait = f" Try again in {max(1, round(retry_after))}s." if retry_after is not None else ""
        super().__init__(f"Rate limit reached: {limit} requests per minute.{wait}")
        self.limit = limit
        self.retry_after = retry_after


class RateLimiter:
    """Per-key quotas, enforced with token buckets in Redis."""

    def __init__(
        self,
        store: BucketStore,
        *,
        default_rpm: int | None = None,
        now_ms: Callable[[], int] | None = None,
    ) -> None:
        self._store = store
        self._default_rpm = default_rpm
        self._now_ms = now_ms  # tests only; production uses Redis's clock

    def rpm_for(self, auth: AuthContext) -> int | None:
        return auth.rpm_limit if auth.rpm_limit is not None else self._default_rpm

    async def check_request(self, auth: AuthContext) -> None:
        """Count one request against the key's per-minute limit; raise if it is used up.

        The bucket holds a full minute's allowance and refills continuously, so a quiet key can
        burst up to its limit and a busy one is paced to limit/60 requests a second.
        """
        limit = self.rpm_for(auth)
        if limit is None:
            return
        try:
            result = await self._store.take(
                f"rpm:{auth.key_id}",
                cost=1,
                capacity=limit,
                refill_per_second=limit / _SECONDS_PER_MINUTE,
                now_ms=None if self._now_ms is None else self._now_ms(),
            )
        except (RedisError, OSError, TimeoutError):
            # Interim policy: a broken limiter must not take the whole gateway down with it.
            # A configurable fail-open/closed choice replaces this once the quotas are complete.
            logger.warning("rate limiting unavailable, allowing request", exc_info=True)
            return
        if not result.allowed:
            raise RateLimitExceededError(limit, result.retry_after)
