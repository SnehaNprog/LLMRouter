import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from app.core.model_config import ModelConfig
from app.providers.errors import AuthFailed, ProviderDown, Timeout
from app.resilience.breaker import CircuitBreaker, CircuitState, Permit
from app.router.deployment import Deployment

Observer = Callable[[str, CircuitState, CircuitState], None]


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

    def __init__(self, deployment: Permit, provider: Permit) -> None:
        self._deployment = deployment
        self._provider = provider

    def success(self) -> None:
        self._deployment.success()
        self._provider.success()

    def failure(self, error: BaseException) -> None:
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
        self._deployment.release()
        self._provider.release()


class DeploymentUnavailable(ProviderDown):
    """A call was refused without being made, because a circuit breaker is open."""

    def __init__(self, message: str, *, breaker: str, retry_after: float) -> None:
        super().__init__(message)
        self.breaker = breaker
        self.retry_after = retry_after


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


class BreakerBoard:
    """Circuit breakers for every deployment and every provider endpoint, made on first use.

    A deployment's breaker notices one model or key going bad; its provider's breaker notices
    the whole vendor going down. A call needs both to agree.
    """

    def __init__(
        self,
        settings: BreakerSettings | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        on_transition: Observer | None = None,
    ) -> None:
        self._settings = settings or BreakerSettings()
        self._clock = clock
        self._on_transition = on_transition
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
            on_transition=self._on_transition,
        )

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

    def acquire(self, deployment: Deployment) -> DeploymentPermit:
        """Permission to call a deployment, or DeploymentUnavailable if either breaker refuses."""
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
        return DeploymentPermit(own_permit, vendor_permit)

    def is_open(self, deployment: Deployment) -> bool:
        """Whether a call right now would be refused (ignoring trial slots already taken)."""
        return (
            self.deployment_breaker(deployment).state is CircuitState.OPEN
            or self.provider_breaker(deployment).state is CircuitState.OPEN
        )

    def states(self) -> dict[str, str]:
        """Every breaker made so far and its state, for inspection."""
        both = {**self._deployments, **self._providers}
        return {b.name: b.state.value for b in both.values()}
