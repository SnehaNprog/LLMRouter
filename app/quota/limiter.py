import logging
from collections.abc import Callable

from app.auth.dependencies import AuthContext
from app.quota.bucket import BucketStore
from app.quota.guard import QuotaGuard
from app.quota.headers import Window
from app.quota.reservation import NO_RESERVATION, Reservation
from app.router.tokens import estimate_prompt_tokens
from app.schemas.chat import ChatCompletionRequest

logger = logging.getLogger(__name__)

_SECONDS_PER_MINUTE = 60


class RateLimitExceededError(Exception):
    def __init__(self, limit: int, retry_after: float | None, window: Window) -> None:
        wait = f" Try again in {max(1, round(retry_after))}s." if retry_after is not None else ""
        super().__init__(f"Rate limit reached: {limit} requests per minute.{wait}")
        self.limit = limit
        self.retry_after = retry_after
        self.window = window


class TokenLimitExceededError(Exception):
    """A key's tokens-per-minute allowance cannot cover a request right now.

    With no ``retry_after`` the request is simply too big for the key's limit, so waiting would
    never help and the caller should shrink it.
    """

    def __init__(
        self, limit: int, requested: int, retry_after: float | None, window: Window
    ) -> None:
        self.limit = limit
        self.window = window
        self.requested = requested
        self.retry_after = retry_after
        self.too_large = retry_after is None
        if self.too_large:
            message = (
                f"Request too large: it needs about {requested} tokens, more than this key's "
                f"limit of {limit} tokens per minute. Reduce the prompt or max_tokens."
            )
        else:
            message = (
                f"Rate limit reached: {limit} tokens per minute "
                f"(this request needs about {requested}). "
                f"Try again in {max(1, round(retry_after or 0))}s."
            )
        super().__init__(message)


class RateLimiter:
    """Per-key quotas, enforced with token buckets in Redis."""

    def __init__(
        self,
        store: BucketStore,
        *,
        default_rpm: int | None = None,
        default_tpm: int | None = None,
        completion_reserve: int = 512,
        now_ms: Callable[[], int] | None = None,
        guard: QuotaGuard | None = None,
    ) -> None:
        self._store = store
        self._guard = guard or QuotaGuard()
        self._default_rpm = default_rpm
        self._default_tpm = default_tpm
        self._completion_reserve = completion_reserve
        self._now_ms = now_ms  # tests only; production uses Redis's clock

    def rpm_for(self, auth: AuthContext) -> int | None:
        return auth.rpm_limit if auth.rpm_limit is not None else self._default_rpm

    def tpm_for(self, auth: AuthContext) -> int | None:
        return auth.tpm_limit if auth.tpm_limit is not None else self._default_tpm

    def _clock(self) -> int | None:
        return None if self._now_ms is None else self._now_ms()

    async def check_request(self, auth: AuthContext) -> Window | None:
        """Count one request against the key's per-minute limit; raise if it is used up.

        Returns where the allowance now stands (None when there is no limit to report on).

        The bucket holds a full minute's allowance and refills continuously, so a quiet key can
        burst up to its limit and a busy one is paced to limit/60 requests a second.
        """
        limit = self.rpm_for(auth)
        if limit is None:
            return None
        result = await self._guard.enforce(
            lambda: self._store.take(
                f"rpm:{auth.key_id}",
                cost=1,
                capacity=limit,
                refill_per_second=limit / _SECONDS_PER_MINUTE,
                now_ms=self._clock(),
            ),
            None,
            what="rate limiting",
        )
        if result is None:
            return None  # Redis is unavailable and the policy is to fail open
        window = Window.from_balance(limit, result.remaining)
        if not result.allowed:
            raise RateLimitExceededError(limit, result.retry_after, window)
        return window

    def estimate_tokens(self, request: ChatCompletionRequest) -> tuple[int, int]:
        """(prompt tokens, tokens to reserve) for a request, before anything is known for sure."""
        prompt = estimate_prompt_tokens(request)
        completion = (request.max_tokens or self._completion_reserve) * max(1, request.n)
        return prompt, prompt + completion

    async def reserve_tokens(
        self, auth: AuthContext, request: ChatCompletionRequest
    ) -> Reservation:
        """Set aside the tokens a request will probably need; raise if the key cannot cover them.

        The real usage is only known once the model has answered, so the estimate (the prompt
        plus ``max_tokens``, or a default) is taken up front and the difference is settled
        afterwards. Without this a key could start any number of requests at once.
        """
        limit = self.tpm_for(auth)
        if limit is None:
            return NO_RESERVATION
        prompt, needed = self.estimate_tokens(request)
        name = f"tpm:{auth.key_id}"
        rate = limit / _SECONDS_PER_MINUTE
        result = await self._guard.enforce(
            lambda: self._store.take(
                name, cost=needed, capacity=limit, refill_per_second=rate, now_ms=self._clock()
            ),
            None,
            what="token limiting",
        )
        if result is None:
            return NO_RESERVATION  # Redis is unavailable and the policy is to fail open
        if not result.allowed:
            window = Window.from_balance(limit, result.remaining)
            raise TokenLimitExceededError(limit, needed, result.retry_after, window)

        async def adjust(delta: int) -> int | None:
            # If this fails the reservation simply stands: charged the estimate, never less.
            return await self._guard.settle(
                lambda: self._store.adjust(
                    name, delta=delta, capacity=limit, refill_per_second=rate, now_ms=self._clock()
                ),
                what="token reservation",
            )

        return Reservation(needed, prompt, adjust, balance=result.remaining)

    def token_window(self, auth: AuthContext, reservation: Reservation) -> Window | None:
        """Where the key's token allowance stands, as of the reservation's latest balance."""
        limit = self.tpm_for(auth)
        if limit is None or reservation.balance is None:
            return None
        return Window.from_balance(limit, reservation.balance)
