import logging
import time
from collections.abc import Callable
from enum import StrEnum

from app.resilience.window import RollingCounter

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

    def __init__(self, breaker: "CircuitBreaker", generation: int, probe: int | None) -> None:
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

    * **closed**: calls flow. The breaker opens when ``failure_threshold`` calls in a row fail,
      or when at least ``min_requests`` calls happened in the last ``window_seconds`` and the
      share that failed reached ``failure_rate_threshold``. The rate rule catches a dependency
      that fails often but never several times running; ``min_requests`` stops one early
      failure out of two calls from looking like a 50% failure rate. The rate is judged when a
      call fails; a success is never what opens the breaker. Either rule can be switched off
      by passing None, but not both.
    * **open**: calls are refused immediately, so a struggling dependency is left alone and
      callers fail fast instead of waiting on timeouts. After ``cooldown`` seconds it moves on.
    * **half-open**: at most ``half_open_max_calls`` trial calls are let through at a time.
      ``half_open_successes`` trials must succeed to close the breaker (one by default); a
      trial that fails opens it again at once, with a fresh cooldown. A trial that is never
      reported on (its caller crashed, say) would hold its slot for ever, so after
      ``probe_timeout`` seconds the slot is taken back and a late report from it is ignored.

    Every state change starts a new *generation*. A call remembers the generation it was
    permitted in, and a report from an older one is ignored for deciding the state: a straggler
    that fails after the breaker has already opened (and reopened) says nothing about whether
    the dependency has recovered now.
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int | None = 5,
        failure_rate_threshold: float | None = None,
        min_requests: int = 20,
        window_seconds: float = 60.0,
        cooldown: float = 30.0,
        half_open_max_calls: int = 1,
        half_open_successes: int = 1,
        probe_timeout: float | None = 30.0,
        clock: Callable[[], float] = time.monotonic,
        on_transition: Callable[[str, CircuitState, CircuitState], None] | None = None,
    ) -> None:
        if failure_threshold is not None and failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if failure_rate_threshold is not None and not 0 < failure_rate_threshold <= 1:
            raise ValueError("failure_rate_threshold must be above 0 and at most 1")
        if failure_threshold is None and failure_rate_threshold is None:
            raise ValueError("set failure_threshold, failure_rate_threshold or both")
        if min_requests < 1:
            raise ValueError("min_requests must be at least 1")
        if cooldown < 0:
            raise ValueError("cooldown must not be negative")
        if half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be at least 1")
        if half_open_successes < 1:
            raise ValueError("half_open_successes must be at least 1")
        if probe_timeout is not None and probe_timeout <= 0:
            raise ValueError("probe_timeout must be positive (or None for no limit)")
        self.name = name
        self._threshold = failure_threshold
        self._rate_threshold = failure_rate_threshold
        self._min_requests = min_requests
        self._window = RollingCounter(window_seconds, clock=clock)
        self.last_trip_reason: str | None = None
        self._cooldown = cooldown
        self._max_probes = half_open_max_calls
        self._needed_successes = half_open_successes
        self._probe_timeout = probe_timeout
        self._clock = clock
        self._on_transition = on_transition
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._probes: dict[int, float] = {}  # trial calls in flight: id -> when they started
        self._next_probe = 0
        self._probe_successes = 0
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
    def failure_rate(self) -> float:
        """Share of calls that failed in the window (0 when there were none)."""
        return self._window.rate

    @property
    def window_requests(self) -> int:
        return self._window.total

    @property
    def probes_in_flight(self) -> int:
        self._reap_probes()
        return len(self._probes)

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
            return Permit(self, self._generation, probe=None)
        if self._state is CircuitState.HALF_OPEN:
            self._reap_probes()
            if len(self._probes) < self._max_probes:
                self._next_probe += 1
                self._probes[self._next_probe] = self._clock()
                return Permit(self, self._generation, probe=self._next_probe)
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
        self._probes.clear()
        self._probe_successes = 0
        self._window.clear()

    def trip(self, remaining_cooldown: float, reason: str) -> bool:
        """Open the breaker because something outside it knows the dependency is down.

        It stays open for ``remaining_cooldown`` seconds rather than a full cooldown, so a breaker
        told about an outage that began a while ago does not outlast it. Does nothing, and returns
        False, if the breaker is already open.
        """
        if self._state is CircuitState.OPEN:
            return False
        self.last_trip_reason = reason
        self._transition(CircuitState.OPEN)
        self._opened_at = self._clock() - (self._cooldown - remaining_cooldown)
        return True

    # ---- the state machine ----

    def _reap_probes(self) -> None:
        """Take back the slots of trial calls that were never reported on."""
        if self._probe_timeout is None or not self._probes:
            return
        now = self._clock()
        for probe, started in list(self._probes.items()):
            if now - started >= self._probe_timeout:
                del self._probes[probe]
                logger.warning(
                    "circuit breaker %s: a trial call was never reported on; its slot is free",
                    self.name,
                    extra={"breaker": self.name},
                )

    def _refresh(self) -> None:
        if self._state is CircuitState.OPEN and self._clock() - self._opened_at >= self._cooldown:
            self._transition(CircuitState.HALF_OPEN)

    def _transition(self, new: CircuitState) -> None:
        old, self._state = self._state, new
        self._generation += 1  # calls permitted before this moment no longer steer the breaker
        self._probes.clear()
        self._probe_successes = 0
        self._window.clear()  # a new episode starts from a clean record
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
            extra={
                "breaker": self.name,
                "from_state": old.value,
                "to_state": new.value,
                "reason": self.last_trip_reason if new is CircuitState.OPEN else None,
            },
        )
        if self._on_transition is not None:
            try:
                self._on_transition(self.name, old, new)
            except Exception:  # an observer's bug must never break the calls being protected
                logger.exception("circuit breaker %s: transition observer failed", self.name)

    def _current(self, generation: int) -> bool:
        self._refresh()
        return generation == self._generation

    def _finish_probe(self, probe: int | None) -> bool:
        """Free a trial call's slot; False if it is not (or no longer) a trial in flight."""
        self._reap_probes()
        return probe is not None and self._probes.pop(probe, None) is not None

    def _on_success(self, generation: int, probe: int | None) -> None:
        if not self._current(generation):
            return
        if self._state is CircuitState.CLOSED:
            self._failures = 0
            self._window.record(failed=False)
        elif self._state is CircuitState.HALF_OPEN and self._finish_probe(probe):
            self._probe_successes += 1
            if self._probe_successes >= self._needed_successes:
                self._transition(CircuitState.CLOSED)

    def _on_failure(self, generation: int, probe: int | None) -> None:
        if not self._current(generation):
            return
        if self._state is CircuitState.CLOSED:
            self._failures += 1
            self._window.record(failed=True)
            reason = self._trip_reason()
            if reason is not None:
                self.last_trip_reason = reason
                self._transition(CircuitState.OPEN)
        elif self._state is CircuitState.HALF_OPEN and self._finish_probe(probe):
            self.last_trip_reason = "a trial call failed"
            self._transition(CircuitState.OPEN)

    def _trip_reason(self) -> str | None:
        if self._threshold is not None and self._failures >= self._threshold:
            return f"{self._failures} consecutive failures"
        if (
            self._rate_threshold is not None
            and self._window.total >= self._min_requests
            and self._window.rate >= self._rate_threshold
        ):
            return (
                f"{self._window.failures} of {self._window.total} recent calls failed "
                f"({self._window.rate:.0%})"
            )
        return None

    def _on_release(self, generation: int, probe: int | None) -> None:
        if self._current(generation) and self._state is CircuitState.HALF_OPEN:
            self._finish_probe(probe)
