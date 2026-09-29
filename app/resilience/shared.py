import logging
import math
import time
from collections.abc import Callable

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class SharedBreakerStore:
    """Tells every gateway replica which circuit breakers are open, through Redis.

    A breaker that opens on one replica leaves a key behind that expires when its cooldown ends,
    so the other replicas learn that the dependency is down without each having to fail against
    it first. The key's own time-to-live carries the remaining cooldown, so replicas never
    compare clocks. A breaker that closes again deletes the key.

    Redis is only an accelerator. Every breaker keeps working from its own observations, so when
    Redis is unreachable the store reports "nothing known" and stays out of the way for
    ``retry_after`` seconds instead of making every request wait on a hung connection.
    """

    def __init__(
        self,
        redis: Redis,
        prefix: str = "llmrouter:breaker:",
        *,
        retry_after: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._redis = redis
        self._prefix = prefix
        self._retry_after = retry_after
        self._clock = clock
        self._down_until = 0.0

    @property
    def available(self) -> bool:
        """False while the store is keeping away from a Redis that just failed."""
        return self._clock() >= self._down_until

    def _failed(self, action: str, name: str, error: Exception) -> None:
        if self.available:  # log the first failure of a spell, not every skipped call after it
            logger.warning(
                "shared circuit breaker state unavailable (%s)",
                type(error).__name__,
                extra={"action": action, "breaker": name},
            )
        self._down_until = self._clock() + self._retry_after

    async def publish_open(self, name: str, seconds: float) -> bool:
        """Record that a breaker is open for the next ``seconds``; False if Redis refused."""
        if not self.available or seconds <= 0:
            return False
        try:
            await self._redis.set(self._prefix + name, "open", px=max(1, math.ceil(seconds * 1000)))
        except Exception as exc:
            self._failed("publish_open", name, exc)
            return False
        return True

    async def publish_closed(self, name: str) -> bool:
        if not self.available:
            return False
        try:
            await self._redis.delete(self._prefix + name)
        except Exception as exc:
            self._failed("publish_closed", name, exc)
            return False
        return True

    async def remaining_open(self, name: str) -> float | None:
        """Seconds left on the breaker's cooldown if some replica has it open, else None."""
        if not self.available:
            return None
        try:
            milliseconds = int(await self._redis.pttl(self._prefix + name))
        except Exception as exc:
            self._failed("remaining_open", name, exc)
            return None
        return milliseconds / 1000 if milliseconds > 0 else None
