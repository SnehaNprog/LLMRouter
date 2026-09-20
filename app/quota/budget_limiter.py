import logging
import math
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from app.auth.dependencies import AuthContext
from app.quota.budget import (
    BudgetStore,
    Counter,
    cost_nano,
    month_period,
    nano_to_usd,
    seconds_until_next_month,
)
from app.quota.guard import QuotaGuard
from app.quota.reservation import NO_RESERVATION, Reservation
from app.router.registry import ModelRegistry
from app.schemas.chat import ChatCompletionRequest

logger = logging.getLogger(__name__)

_GRACE_SECONDS = 86_400  # a counter outlives its month by a day, so late settlements still land
_PER_MILLION = 1_000_000


def _money(usd: float) -> str:
    return f"${usd:,.2f}" if usd >= 1 else f"${usd:.6f}".rstrip("0")


class BudgetExceededError(Exception):
    """A monthly spending cap has been reached by a key or by its whole organisation."""

    def __init__(
        self, scope: Literal["key", "tenant"], limit_nano: int, retry_after: float
    ) -> None:
        subject = "API key" if scope == "key" else "organisation"
        days = max(1, math.ceil(retry_after / 86_400))
        super().__init__(
            f"The monthly budget of {_money(nano_to_usd(limit_nano))} for this {subject} "
            f"is used up. It resets at the start of next month (in about {days} days)."
        )
        self.scope = scope
        self.limit_nano = limit_nano
        self.retry_after = retry_after


class BudgetLimiter:
    """Monthly spending caps per key and per tenant, enforced with atomic Redis counters."""

    def __init__(
        self,
        store: BudgetStore,
        *,
        completion_reserve: int = 512,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        guard: QuotaGuard | None = None,
    ) -> None:
        self._store = store
        self._guard = guard or QuotaGuard()
        self._completion_reserve = completion_reserve
        self._clock = clock

    def _counters(self, auth: AuthContext, now: datetime) -> list[Counter]:
        period = month_period(now)
        ttl = seconds_until_next_month(now) + _GRACE_SECONDS
        counters = []
        if auth.key_budget_nano is not None:
            counters.append(Counter(f"key:{auth.key_id}:{period}", auth.key_budget_nano, ttl))
        if auth.tenant_budget_nano is not None:
            counters.append(
                Counter(f"tenant:{auth.tenant_id}:{period}", auth.tenant_budget_nano, ttl)
            )
        return counters

    def estimate_cost(
        self,
        registry: ModelRegistry,
        request: ChatCompletionRequest,
        prompt_tokens: int,
    ) -> int:
        """The most this request could cost, in nano-dollars, over everything that might serve it.

        Which deployment answers is only decided by the router (retries and fallbacks included),
        so the estimate assumes the priciest model in the request's fallback chain.
        """
        deployments = [d for name in registry.chain(request.model) for d in registry.get(name)]
        dearest_in = max(d.config.input_cost_per_mtok for d in deployments)
        dearest_out = max(d.config.output_cost_per_mtok for d in deployments)
        completion = (request.max_tokens or self._completion_reserve) * max(1, request.n)
        usd = (prompt_tokens * dearest_in + completion * dearest_out) / _PER_MILLION
        return cost_nano(usd)

    async def reserve(
        self,
        auth: AuthContext,
        request: ChatCompletionRequest,
        registry: ModelRegistry,
        prompt_tokens: int,
    ) -> Reservation:
        """Count the request's worst-case cost against the budgets, or raise if one is used up.

        The real cost is only known afterwards, so this is settled against it: the key and
        tenant counters are charged the estimate now (all or nothing) and corrected later.
        """
        now = self._clock()
        counters = self._counters(auth, now)
        if not counters:
            return NO_RESERVATION
        needed = self.estimate_cost(registry, request, prompt_tokens)
        if needed == 0:
            return NO_RESERVATION  # free models cannot overspend anything
        result = await self._guard.enforce(
            lambda: self._store.spend(counters, needed), None, what="budget limiting"
        )
        if result is None:
            return NO_RESERVATION  # Redis is unavailable and the policy is to fail open
        if not result.allowed:
            blocked = next(c for c in counters if c.name == result.blocked_by)
            scope: Literal["key", "tenant"] = "key" if blocked.name.startswith("key:") else "tenant"
            raise BudgetExceededError(scope, blocked.limit, seconds_until_next_month(now))

        async def adjust(delta: int) -> int | None:
            # The reservation was charged up front; settling corrects it to the real cost.
            await self._guard.settle(
                lambda: self._store.adjust(counters, -delta), what="budget reservation"
            )
            return None

        return Reservation(needed, 0, adjust)
