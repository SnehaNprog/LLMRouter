import logging
import time
from collections.abc import Callable
from enum import StrEnum

logger = logging.getLogger("app.resilience.breaker")


class CircuitState(StrEnum):
    CLOSED = "closed"  # healthy: calls flow, failures are counted
    OPEN = "open"  # failing: calls are refused without being tried
    HALF_OPEN = "half_open"  # recovering: a few trial calls decide whether to resume


class CircuitOpenError(Exception):
    """A call was refused because the breaker is open (or its trial slots are taken)."""

    def __init__(self, name: str, retry_after: float) -> None:
        super().__init__(f"circuit '{name}' is open")
        self.name = name
        self.retry_after = retry_after


class Permit:
    """Permission to make one call. Report how it went with exactly one of the three methods.

    Only the first report counts, so a cleanup path can call ``release()`` without worrying that
    the call was already reported. A call that is abandoned (cancelled, say) should be released:
    that frees a trial slot without counting for or against the dependency.
    """

    def __init__(self, breaker: "CircuitBreaker", generation: int, probe: bool) -> None:
        self._breaker = breaker
        self._generation = generation
        self._probe = probe
        self._reported = False

    def success(self) -> None:
        if not self._reported:
            self._reported = True
            self._breaker._on_success(self._generation, self._probe)

    def failure(self) -> None:
        if not self._reported:
            self._reported = True
            self._breaker._on_failure(self._generation, self._probe)

    def release(self) -> None:
        if not self._reported:
            self._reported = True
            self._breaker._on_release(self._generation, self._probe)


class CircuitBreaker:
    """Stops calling something that keeps failing, and checks cautiously whether it has recovered.

    * **closed**: calls flow. ``failure_threshold`` failures in a row open the breaker; any
      success resets the count.
    * **open**: calls are refused immediately, so a struggling dependency is left alone and
      callers fail fast instead of waiting on timeouts. After ``cooldown`` seconds it moves on.
    * **half-open**: at most ``half_open_max_calls`` trial calls are let through at a time. A
      trial that succeeds closes the breaker; a trial that fails opens it again with a fresh
      cooldown.

    Every state change starts a new *generation*. A call remembers the generation it was
    permitted in, and a report from an older one is ignored for deciding the state: a straggler
    that fails after the breaker has already opened (and reopened) says nothing about whether
    the dependency has recovered now.
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        cooldown: float = 30.0,
        half_open_max_calls: int = 1,
        clock: Callable[[], float] = time.monotonic,
        on_transition: Callable[[str, CircuitState, CircuitState], None] | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if cooldown < 0:
            raise ValueError("cooldown must not be negative")
        if half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be at least 1")
        self.name = name
        self._threshold = failure_threshold
        self._cooldown = cooldown
        self._max_probes = half_open_max_calls
        self._clock = clock
        self._on_transition = on_transition
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._probes = 0
        self._opened_at = 0.0
        self._generation = 0

    # ---- reading the breaker ----

    @property
    def state(self) -> CircuitState:
        self._refresh()
        return self._state

    @property
    def consecutive_failures(self) -> int:
        return self._failures

    @property
    def probes_in_flight(self) -> int:
        return self._probes

    def remaining_cooldown(self) -> float:
        """Seconds until an open breaker will let a trial call through (0 if it is not open)."""
        self._refresh()
        if self._state is not CircuitState.OPEN:
            return 0.0
        return max(0.0, self._opened_at + self._cooldown - self._clock())

    # ---- asking permission ----

    def try_acquire(self) -> Permit | None:
        """Permission for one call, or None if the breaker refuses it."""
        self._refresh()
        if self._state is CircuitState.CLOSED:
            return Permit(self, self._generation, probe=False)
        if self._state is CircuitState.HALF_OPEN and self._probes < self._max_probes:
            self._probes += 1
            return Permit(self, self._generation, probe=True)
        return None

    def acquire(self) -> Permit:
        """Like ``try_acquire``, but raises CircuitOpenError when refused."""
        permit = self.try_acquire()
        if permit is None:
            raise CircuitOpenError(self.name, self.remaining_cooldown())
        return permit

    def reset(self) -> None:
        """Close the breaker and forget every failure: a clean slate (an operator's override).

        Calls already in flight are forgotten too: whatever they report afterwards is ignored.
        """
        if self._state is not CircuitState.CLOSED:
            self._transition(CircuitState.CLOSED)
        else:
            self._generation += 1
        self._failures = 0
        self._probes = 0

    # ---- the state machine ----

    def _refresh(self) -> None:
        if self._state is CircuitState.OPEN and self._clock() - self._opened_at >= self._cooldown:
            self._transition(CircuitState.HALF_OPEN)

    def _transition(self, new: CircuitState) -> None:
        old, self._state = self._state, new
        self._generation += 1  # calls permitted before this moment no longer steer the breaker
        self._probes = 0
        if new is CircuitState.OPEN:
            self._opened_at = self._clock()
        elif new is CircuitState.CLOSED:
            self._failures = 0
        level = logging.WARNING if new is CircuitState.OPEN else logging.INFO
        logger.log(
            level,
            "circuit breaker %s: %s -> %s",
            self.name,
            old.value,
            new.value,
            extra={"breaker": self.name, "from_state": old.value, "to_state": new.value},
        )
        if self._on_transition is not None:
            try:
                self._on_transition(self.name, old, new)
            except Exception:  # an observer's bug must never break the calls being protected
                logger.exception("circuit breaker %s: transition observer failed", self.name)

    def _current(self, generation: int) -> bool:
        self._refresh()
        return generation == self._generation

    def _on_success(self, generation: int, probe: bool) -> None:
        if not self._current(generation):
            return
        if self._state is CircuitState.CLOSED:
            self._failures = 0
        elif self._state is CircuitState.HALF_OPEN and probe:
            self._transition(CircuitState.CLOSED)

    def _on_failure(self, generation: int, probe: bool) -> None:
        if not self._current(generation):
            return
        if self._state is CircuitState.CLOSED:
            self._failures += 1
            if self._failures >= self._threshold:
                self._transition(CircuitState.OPEN)
        elif self._state is CircuitState.HALF_OPEN and probe:
            self._transition(CircuitState.OPEN)

    def _on_release(self, generation: int, probe: bool) -> None:
        if probe and self._current(generation) and self._state is CircuitState.HALF_OPEN:
            self._probes = max(0, self._probes - 1)
