import asyncio
import logging
import math
import time
from collections.abc import Callable, Coroutine, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.core.model_config import ModelConfig
from app.providers.errors import AuthFailed, ProviderDown, RateLimited, Timeout
from app.resilience.breaker import CircuitBreaker, CircuitState, Permit
from app.resilience.shared import SharedBreakerStore
from app.router.deployment import Deployment

Observer = Callable[[str, CircuitState, CircuitState], None]
logger = logging.getLogger(__name__)


def provider_key(config: ModelConfig) -> str:
    """What deployments share when one outage takes them all down: the vendor's endpoint.

    Two models (or two API keys) on the same endpoint share it, so a vendor outage opens one
    breaker for all of them rather than each having to find out for itself.
    """
    return f"{config.provider}@{config.base_url or 'default'}"


class Health(Enum):
    """What a failed call says about the thing that was called."""

    PROVIDER_DOWN = "provider_down"  # the vendor is failing: counts against it and the deployment
    DEPLOYMENT_BAD = "deployment_bad"  # only this deployment (its key, say) is at fault
    NEUTRAL = "neutral"  # says nothing about health: a bad request, our own bug, a cancellation


def health_of(error: BaseException) -> Health:
    """Whether a failure should count against a breaker.

    Only errors that show an upstream is unhealthy count. A request the vendor rejected as
    invalid says nothing about the vendor, and a 429 is the vendor working as designed (how
    to treat throttling is decided separately), so neither can push a breaker open.
    """
    if isinstance(error, ProviderDown | Timeout):
        return Health.PROVIDER_DOWN
    if isinstance(error, AuthFailed):
        return Health.DEPLOYMENT_BAD  # a rejected key says nothing about the vendor's other keys
    return Health.NEUTRAL


class DeploymentPermit:
    """Permission to call one deployment: a permit from its breaker and one from its provider's.

    Report with exactly one of ``success``, ``failure(error)`` or ``release``; later reports
    are ignored.
    """

    def __init__(
        self,
        deployment: Permit,
        provider: Permit,
        *,
        on_success: Callable[[], None] = lambda: None,
        on_throttled: Callable[[float | None], object] = lambda retry_after: None,
    ) -> None:
        self._deployment = deployment
        self._provider = provider
        self._on_success = on_success
        self._on_throttled = on_throttled
        self._reported = False

    def success(self) -> None:
        if self._reported:
            return
        self._reported = True
        self._deployment.success()
        self._provider.success()
        self._on_success()

    def failure(self, error: BaseException) -> None:
        if self._reported:
            return
        if isinstance(error, RateLimited):
            # The vendor is up and working as designed, so no breaker hears of it; this
            # deployment is just given a rest for as long as the vendor asked.
            self.release()
            self._on_throttled(error.retry_after)
            return
        self._reported = True
        match health_of(error):
            case Health.PROVIDER_DOWN:
                self._deployment.failure()
                self._provider.failure()
            case Health.DEPLOYMENT_BAD:
                self._deployment.failure()
                self._provider.release()
            case Health.NEUTRAL:
                self.release()

    def release(self) -> None:
        self._reported = True
        self._deployment.release()
        self._provider.release()


class DeploymentUnavailable(ProviderDown):
    """A call was refused without being made, because a circuit breaker is open."""

    def __init__(self, message: str, *, breaker: str, retry_after: float) -> None:
        super().__init__(message)
        self.breaker = breaker
        self.retry_after = retry_after


class DeploymentThrottled(RateLimited):
    """A call was refused without being made, because the vendor recently said to slow down."""

    def __init__(self, message: str, *, retry_after: float) -> None:
        super().__init__(message, retry_after=retry_after)


@dataclass(frozen=True)
class BreakerSettings:
    failure_threshold: int | None = 5
    failure_rate_threshold: float | None = 0.5
    min_requests: int = 20
    window_seconds: float = 60.0
    cooldown: float = 30.0
    half_open_max_calls: int = 1
    half_open_successes: int = 1
    probe_timeout: float | None = 30.0
    # How long a deployment is left alone after a 429 that says nothing about when to return;
    # it doubles with every further 429 in a row. A Retry-After from the vendor is used as given.
    throttle_cooldown: float = 5.0
    throttle_max_cooldown: float = 60.0  # the longest any 429 can keep a deployment out


class BreakerBoard:
    """Circuit breakers for every deployment and every provider endpoint, made on first use.

    A deployment's breaker notices one model or key going bad; its provider's breaker notices
    the whole vendor going down. A call needs both to agree.

    With a ``shared`` store, breakers that open are announced to the other gateway replicas, and
    ``sync`` opens the local breakers that another replica has opened. Each breaker is looked up
    at most once per ``sync_interval`` seconds, so the shared state costs a Redis round trip only
    now and then, not on every request.
    """

    def __init__(
        self,
        settings: BreakerSettings | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        on_transition: Observer | None = None,
        shared: SharedBreakerStore | None = None,
        sync_interval: float = 1.0,
        on_throttle: Callable[[Deployment, float], None] | None = None,
    ) -> None:
        self._settings = settings or BreakerSettings()
        self._clock = clock
        self._on_transition = on_transition
        self._shared = shared
        self._sync_interval = sync_interval
        self._on_throttle = on_throttle
        self._synced_at: dict[str, float] = {}
        self._applying_shared = False  # True while opening a breaker on another replica's say-so
        self._pending: set[asyncio.Task[object]] = set()
        self._throttled_until: dict[str, float] = {}
        self._throttle_streak: dict[str, int] = {}
        self._deployments: dict[str, CircuitBreaker] = {}
        self._providers: dict[str, CircuitBreaker] = {}

    def _make(self, name: str) -> CircuitBreaker:
        s = self._settings
        return CircuitBreaker(
            name,
            failure_threshold=s.failure_threshold,
            failure_rate_threshold=s.failure_rate_threshold,
            min_requests=s.min_requests,
            window_seconds=s.window_seconds,
            cooldown=s.cooldown,
            half_open_max_calls=s.half_open_max_calls,
            half_open_successes=s.half_open_successes,
            probe_timeout=s.probe_timeout,
            clock=self._clock,
            on_transition=self._observe,
        )

    def _observe(self, name: str, old: CircuitState, new: CircuitState) -> None:
        if self._on_transition is not None:
            self._on_transition(name, old, new)
        if self._shared is None or self._applying_shared:
            return  # a breaker opened by another replica's word must not be announced again
        if new is CircuitState.OPEN:
            self._announce(self._shared.publish_open(name, self._settings.cooldown))
        elif new is CircuitState.CLOSED:
            self._announce(self._shared.publish_closed(name))

    def _announce(self, publication: Coroutine[Any, Any, bool]) -> None:
        """Send an update to Redis in the background: the call being protected never waits on it."""
        try:
            task = asyncio.get_running_loop().create_task(publication)
        except RuntimeError:  # no event loop (the breakers are being driven synchronously)
            publication.close()
            return
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def sync(self, deployment: Deployment) -> None:
        """Open this deployment's breakers if another replica has opened them."""
        if self._shared is None:
            return
        await self._sync_throttle(deployment)
        for breaker in (self.provider_breaker(deployment), self.deployment_breaker(deployment)):
            now = self._clock()
            if now - self._synced_at.get(breaker.name, -math.inf) < self._sync_interval:
                continue
            self._synced_at[breaker.name] = now
            if breaker.state is CircuitState.OPEN:
                continue  # already refusing calls: nothing to learn
            remaining = await self._shared.remaining_open(breaker.name)
            if remaining is None:
                continue
            self._applying_shared = True
            try:
                breaker.trip(remaining, "another gateway replica opened it")
            finally:
                self._applying_shared = False

    async def _sync_throttle(self, deployment: Deployment) -> None:
        assert self._shared is not None
        key = self._throttle_key(deployment)
        now = self._clock()
        if now - self._synced_at.get(key, -math.inf) < self._sync_interval:
            return
        self._synced_at[key] = now
        if self.throttle_remaining(deployment) > 0:
            return
        remaining = await self._shared.remaining_open(key)
        if remaining is not None:
            self._throttled_until[deployment.id] = self._clock() + remaining

    async def drain(self) -> None:
        """Wait for updates still on their way to Redis (at shutdown, and in tests)."""
        while self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)

    def deployment_breaker(self, deployment: Deployment) -> CircuitBreaker:
        key = deployment.id
        if key not in self._deployments:
            self._deployments[key] = self._make(f"deployment:{deployment.alias}:{key}")
        return self._deployments[key]

    def provider_breaker(self, deployment: Deployment) -> CircuitBreaker:
        key = provider_key(deployment.config)
        if key not in self._providers:
            self._providers[key] = self._make(f"provider:{key}")
        return self._providers[key]

    # ---- being told to slow down (HTTP 429) ----

    def throttle(self, deployment: Deployment, retry_after: float | None = None) -> float:
        """Leave a deployment alone after a 429; returns the seconds it will be left alone.

        The vendor's ``Retry-After`` is honoured as given (up to ``throttle_max_cooldown``).
        Without one the wait starts at ``throttle_cooldown`` and doubles with each further 429
        in a row, so a deployment that stays throttled is not probed ever more hopefully.
        """
        s = self._settings
        streak = self._throttle_streak.get(deployment.id, 0) + 1
        self._throttle_streak[deployment.id] = streak
        if retry_after is not None and retry_after > 0:
            seconds = retry_after
        else:
            seconds = s.throttle_cooldown * 2 ** min(streak - 1, 16)
        seconds = min(seconds, s.throttle_max_cooldown)
        until = self._clock() + seconds
        self._throttled_until[deployment.id] = max(
            until, self._throttled_until.get(deployment.id, 0)
        )
        if self._shared is not None:
            self._announce(self._shared.publish_open(self._throttle_key(deployment), seconds))
        if self._on_throttle is not None:
            try:
                self._on_throttle(deployment, seconds)
            except Exception:  # an observer's bug must never break the calls being protected
                logger.exception("throttle observer failed")
        logger.warning(
            "deployment throttled by its vendor, leaving it alone",
            extra={"deployment": deployment.id, "cooldown": round(seconds, 3), "streak": streak},
        )
        return seconds

    @staticmethod
    def _throttle_key(deployment: Deployment) -> str:
        return f"throttle:{deployment.id}"

    def throttle_remaining(self, deployment: Deployment) -> float:
        """Seconds until a throttled deployment may be called again (0 if it is not throttled)."""
        until = self._throttled_until.get(deployment.id)
        if until is None:
            return 0.0
        remaining = until - self._clock()
        if remaining <= 0:
            del self._throttled_until[deployment.id]  # the streak stays: it ends with a success
            return 0.0
        return remaining

    def expire_throttle(self, deployment: Deployment) -> None:
        """Forget a throttle whose wait the caller has sat out itself."""
        self._throttled_until.pop(deployment.id, None)

    def _succeeded(self, deployment: Deployment) -> None:
        self._throttle_streak.pop(deployment.id, None)

    def acquire(self, deployment: Deployment) -> DeploymentPermit:
        """Permission to call a deployment.

        Raises DeploymentThrottled if the vendor recently asked for calls to slow down, and
        DeploymentUnavailable if either circuit breaker refuses.
        """
        if (waiting := self.throttle_remaining(deployment)) > 0:
            raise DeploymentThrottled(
                f"{deployment.alias}: the provider asked for requests to slow down",
                retry_after=waiting,
            )
        own = self.deployment_breaker(deployment)
        vendor = self.provider_breaker(deployment)
        own_permit = own.try_acquire()
        if own_permit is None:
            raise DeploymentUnavailable(
                f"{deployment.alias}: circuit breaker is open",
                breaker=own.name,
                retry_after=own.remaining_cooldown(),
            )
        vendor_permit = vendor.try_acquire()
        if vendor_permit is None:
            own_permit.release()  # we never made the call, so it must not use up a trial slot
            raise DeploymentUnavailable(
                f"{deployment.alias}: the provider's circuit breaker is open",
                breaker=vendor.name,
                retry_after=vendor.remaining_cooldown(),
            )
        return DeploymentPermit(
            own_permit,
            vendor_permit,
            on_success=lambda: self._succeeded(deployment),
            on_throttled=lambda retry_after: self.throttle(deployment, retry_after),
        )

    def is_open(self, deployment: Deployment) -> bool:
        """Whether a call right now would be refused (ignoring trial slots already taken)."""
        return (
            self.throttle_remaining(deployment) > 0
            or self.deployment_breaker(deployment).state is CircuitState.OPEN
            or self.provider_breaker(deployment).state is CircuitState.OPEN
        )

    def is_throttled_only(self, deployment: Deployment) -> bool:
        """Refused because of a 429, not because anything is broken."""
        return self.throttle_remaining(deployment) > 0 and not (
            self.deployment_breaker(deployment).state is CircuitState.OPEN
            or self.provider_breaker(deployment).state is CircuitState.OPEN
        )

    def wait_for(self, deployment: Deployment) -> float:
        """Seconds until a refused deployment may be tried again (0 if it is not refused)."""
        return max(
            self.deployment_breaker(deployment).remaining_cooldown(),
            self.provider_breaker(deployment).remaining_cooldown(),
            self.throttle_remaining(deployment),
        )

    async def sync_all(self, deployments: Iterable[Deployment]) -> None:
        """Bring the breakers of these deployments up to date with what other replicas know."""
        if self._shared is not None:
            await asyncio.gather(*(self.sync(d) for d in deployments))

    def throttled(self) -> dict[str, float]:
        """Seconds still to wait, by deployment id, for each deployment now resting after a 429."""
        remaining = {key: until - self._clock() for key, until in self._throttled_until.items()}
        return {key: seconds for key, seconds in remaining.items() if seconds > 0}

    def states(self) -> dict[str, str]:
        """Every breaker made so far and its state, for inspection."""
        both = {**self._deployments, **self._providers}
        return {b.name: b.state.value for b in both.values()}
