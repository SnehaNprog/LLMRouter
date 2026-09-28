import logging
import random

import pytest

from app.resilience.breaker import CircuitBreaker, CircuitOpenError, CircuitState, Permit

CLOSED, OPEN, HALF_OPEN = CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make(
    threshold: int = 3, cooldown: float = 10.0, probes: int = 1, **kwargs: object
) -> tuple[CircuitBreaker, Clock]:
    clock = Clock()
    breaker = CircuitBreaker(
        "test",
        failure_threshold=threshold,
        cooldown=cooldown,
        half_open_max_calls=probes,
        clock=clock,
        **kwargs,  # type: ignore[arg-type]
    )
    return breaker, clock


def fail(breaker: CircuitBreaker, times: int = 1) -> None:
    for _ in range(times):
        permit = breaker.try_acquire()
        assert permit is not None
        permit.failure()


def trip(breaker: CircuitBreaker, threshold: int = 3) -> None:
    fail(breaker, threshold)
    assert breaker.state is OPEN


def open_and_wait(breaker: CircuitBreaker, clock: Clock, cooldown: float = 10.0) -> None:
    trip(breaker)
    clock.advance(cooldown)
    assert breaker.state is HALF_OPEN


# ---------- closed ----------


def test_a_new_breaker_is_closed_and_lets_calls_through() -> None:
    breaker, _ = make()

    assert breaker.state is CLOSED
    assert breaker.try_acquire() is not None
    assert breaker.consecutive_failures == 0


def test_it_opens_on_exactly_the_threshold_of_consecutive_failures() -> None:
    breaker, _ = make(threshold=3)

    fail(breaker, 2)
    assert breaker.state is CLOSED
    fail(breaker, 1)

    assert breaker.state is OPEN


def test_a_success_resets_the_count() -> None:
    breaker, _ = make(threshold=3)
    fail(breaker, 2)

    permit = breaker.try_acquire()
    assert permit is not None
    permit.success()
    fail(breaker, 2)

    assert breaker.state is CLOSED  # 2 + 2 failures, but never 3 in a row
    assert breaker.consecutive_failures == 2


def test_a_threshold_of_one_opens_on_the_first_failure() -> None:
    breaker, _ = make(threshold=1)

    fail(breaker)

    assert breaker.state is OPEN


def test_closed_calls_can_run_concurrently() -> None:
    breaker, _ = make()

    permits = [breaker.try_acquire() for _ in range(50)]

    assert all(p is not None for p in permits)


# ---------- open ----------


def test_an_open_breaker_refuses_calls() -> None:
    breaker, _ = make()
    trip(breaker)

    assert breaker.try_acquire() is None
    with pytest.raises(CircuitOpenError) as excinfo:
        breaker.acquire()

    assert excinfo.value.name == "test"
    assert excinfo.value.retry_after == pytest.approx(10.0)


def test_the_cooldown_counts_down() -> None:
    breaker, clock = make(cooldown=10)
    trip(breaker)

    clock.advance(4)

    assert breaker.remaining_cooldown() == pytest.approx(6.0)
    assert breaker.try_acquire() is None
    with pytest.raises(CircuitOpenError) as excinfo:
        breaker.acquire()
    assert excinfo.value.retry_after == pytest.approx(6.0)


def test_it_stays_open_until_the_full_cooldown_has_passed() -> None:
    breaker, clock = make(cooldown=10)
    trip(breaker)

    clock.advance(9.999)
    assert breaker.state is OPEN
    clock.advance(0.001)
    assert breaker.state is HALF_OPEN


def test_remaining_cooldown_is_zero_when_not_open() -> None:
    breaker, clock = make()
    assert breaker.remaining_cooldown() == 0.0
    open_and_wait(breaker, clock)
    assert breaker.remaining_cooldown() == 0.0


def test_a_zero_cooldown_goes_straight_to_half_open() -> None:
    breaker, _ = make(cooldown=0)

    fail(breaker, 3)

    assert breaker.state is HALF_OPEN


def test_failures_reported_while_open_change_nothing() -> None:
    breaker, clock = make(cooldown=10)
    straggler = breaker.try_acquire()  # a call started before the breaker opened
    trip(breaker)
    clock.advance(5)

    assert straggler is not None
    straggler.failure()

    assert breaker.remaining_cooldown() == pytest.approx(5.0)  # the cooldown was not restarted


# ---------- half-open ----------


def test_half_open_admits_only_the_configured_number_of_trials() -> None:
    breaker, clock = make(probes=2)
    open_and_wait(breaker, clock)

    first, second, third = breaker.try_acquire(), breaker.try_acquire(), breaker.try_acquire()

    assert first is not None and second is not None
    assert third is None
    assert breaker.probes_in_flight == 2


def test_a_successful_trial_closes_the_breaker() -> None:
    breaker, clock = make()
    open_and_wait(breaker, clock)
    probe = breaker.try_acquire()
    assert probe is not None

    probe.success()

    assert breaker.state is CLOSED
    assert breaker.consecutive_failures == 0
    assert breaker.try_acquire() is not None  # and traffic flows again


def test_a_failed_trial_reopens_with_a_fresh_cooldown() -> None:
    breaker, clock = make(cooldown=10)
    open_and_wait(breaker, clock)
    probe = breaker.try_acquire()
    assert probe is not None

    probe.failure()

    assert breaker.state is OPEN
    assert breaker.remaining_cooldown() == pytest.approx(10.0)
    assert breaker.try_acquire() is None


def test_a_released_trial_frees_its_slot_without_counting_either_way() -> None:
    breaker, clock = make(probes=1)
    open_and_wait(breaker, clock)
    probe = breaker.try_acquire()
    assert probe is not None and breaker.try_acquire() is None

    probe.release()

    assert breaker.state is HALF_OPEN  # neither closed nor reopened
    assert breaker.probes_in_flight == 0
    assert breaker.try_acquire() is not None  # the slot is available again


def test_a_hundred_simultaneous_callers_get_only_the_allowed_trials() -> None:
    breaker, clock = make(probes=3)
    open_and_wait(breaker, clock)

    permits = [breaker.try_acquire() for _ in range(100)]

    assert sum(p is not None for p in permits) == 3


def test_the_breaker_can_go_round_the_whole_cycle_repeatedly() -> None:
    breaker, clock = make(threshold=2, cooldown=5)
    edges: list[tuple[CircuitState, CircuitState]] = []
    breaker._on_transition = lambda _, old, new: edges.append((old, new))

    for _ in range(3):
        fail(breaker, 2)
        clock.advance(5)
        probe = breaker.try_acquire()
        assert probe is not None
        probe.success()

    assert edges == [(CLOSED, OPEN), (OPEN, HALF_OPEN), (HALF_OPEN, CLOSED)] * 3


# ---------- calls that were already in flight ----------


def test_a_late_failure_from_before_the_breaker_opened_cannot_reopen_a_half_open_one() -> None:
    breaker, clock = make(threshold=2, cooldown=10)
    straggler = breaker.try_acquire()  # permitted while closed, still running
    fail(breaker, 2)
    clock.advance(10)
    assert breaker.state is HALF_OPEN

    assert straggler is not None
    straggler.failure()  # arrives now, long after it was permitted

    assert breaker.state is HALF_OPEN  # it says nothing about whether the dependency recovered


def test_a_late_success_from_before_the_breaker_opened_cannot_close_it() -> None:
    breaker, clock = make(threshold=2, cooldown=10)
    straggler = breaker.try_acquire()
    fail(breaker, 2)

    assert straggler is not None
    straggler.success()
    assert breaker.state is OPEN

    clock.advance(10)
    straggler_two = breaker.try_acquire()
    assert straggler_two is not None  # a proper trial
    assert breaker.state is HALF_OPEN


def test_a_late_success_does_not_wipe_the_failures_counted_since() -> None:
    breaker, _ = make(threshold=3)
    old = breaker.try_acquire()
    breaker.reset()  # a new generation begins
    fail(breaker, 2)

    assert old is not None
    old.success()

    assert breaker.consecutive_failures == 2  # the old call's result is not part of this story


def test_a_late_failure_does_not_count_towards_a_new_generation() -> None:
    breaker, _ = make(threshold=2)
    old = breaker.try_acquire()
    breaker.reset()
    fail(breaker, 1)

    assert old is not None
    old.failure()

    assert breaker.state is CLOSED and breaker.consecutive_failures == 1


def test_an_old_trial_released_after_a_state_change_does_not_free_a_slot_it_never_held() -> None:
    breaker, clock = make(probes=1)
    open_and_wait(breaker, clock)
    old_probe = breaker.try_acquire()
    assert old_probe is not None
    breaker.reset()
    fail(breaker, 3)
    clock.advance(10)
    new_probe = breaker.try_acquire()
    assert new_probe is not None

    old_probe.release()

    assert breaker.probes_in_flight == 1  # the new trial still holds its slot


# ---------- permits ----------


def test_only_the_first_report_counts() -> None:
    breaker, _ = make(threshold=2)
    permit = breaker.try_acquire()
    assert permit is not None

    permit.failure()
    permit.failure()
    permit.success()
    permit.release()

    assert breaker.consecutive_failures == 1


def test_a_success_then_a_failure_report_counts_as_the_success() -> None:
    breaker, _ = make(threshold=1)
    permit = breaker.try_acquire()
    assert permit is not None

    permit.success()
    permit.failure()

    assert breaker.state is CLOSED


def test_releasing_a_closed_call_changes_nothing() -> None:
    breaker, _ = make(threshold=2)
    fail(breaker, 1)
    permit = breaker.try_acquire()
    assert permit is not None

    permit.release()

    assert breaker.consecutive_failures == 1 and breaker.state is CLOSED


def test_permits_are_permits() -> None:
    breaker, _ = make()

    assert isinstance(breaker.acquire(), Permit)


# ---------- overriding and observing ----------


def test_reset_closes_an_open_breaker_and_forgets_the_failures() -> None:
    breaker, _ = make()
    trip(breaker)

    breaker.reset()

    assert breaker.state is CLOSED and breaker.consecutive_failures == 0
    assert breaker.try_acquire() is not None


def test_reset_on_a_healthy_breaker_just_forgets_failures() -> None:
    breaker, _ = make(threshold=3)
    fail(breaker, 2)

    breaker.reset()

    assert breaker.consecutive_failures == 0 and breaker.state is CLOSED


def test_observers_hear_about_every_transition() -> None:
    seen: list[tuple[str, CircuitState, CircuitState]] = []
    breaker, clock = make(on_transition=lambda name, old, new: seen.append((name, old, new)))

    trip(breaker)
    clock.advance(10)
    assert breaker.state is HALF_OPEN
    breaker.reset()

    assert seen == [
        ("test", CLOSED, OPEN),
        ("test", OPEN, HALF_OPEN),
        ("test", HALF_OPEN, CLOSED),
    ]


def test_a_broken_observer_cannot_break_the_breaker(caplog: pytest.LogCaptureFixture) -> None:
    def explode(*_: object) -> None:
        raise RuntimeError("observer bug")

    breaker, _ = make(threshold=1, on_transition=explode)

    with caplog.at_level(logging.ERROR, logger="app.resilience.breaker"):
        fail(breaker)

    assert breaker.state is OPEN  # the transition still happened
    assert any("observer failed" in r.getMessage() for r in caplog.records)


def test_transitions_are_logged_with_the_right_severity(caplog: pytest.LogCaptureFixture) -> None:
    breaker, clock = make()

    with caplog.at_level(logging.INFO, logger="app.resilience.breaker"):
        trip(breaker)
        clock.advance(10)
        probe = breaker.try_acquire()
        assert probe is not None
        probe.success()

    levels = [(r.levelname, r.to_state) for r in caplog.records]  # type: ignore[attr-defined]
    assert levels == [("WARNING", "open"), ("INFO", "half_open"), ("INFO", "closed")]
    assert caplog.records[0].breaker == "test"  # type: ignore[attr-defined]


def test_breakers_are_independent_of_each_other() -> None:
    one, _ = make()
    two, _ = make()

    trip(one)

    assert one.state is OPEN and two.state is CLOSED


@pytest.mark.parametrize(
    "kwargs",
    [{"failure_threshold": 0}, {"cooldown": -1}, {"half_open_max_calls": 0}],
)
def test_nonsense_settings_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        CircuitBreaker("x", **kwargs)  # type: ignore[arg-type]


# ---------- invariants over random histories ----------

ALLOWED_EDGES = {
    (CLOSED, OPEN),
    (OPEN, HALF_OPEN),
    (HALF_OPEN, CLOSED),
    (HALF_OPEN, OPEN),
    (OPEN, CLOSED),  # only by an operator's reset()
}


def test_random_histories_never_break_the_rules() -> None:
    """Throw thousands of random calls, reports and clock ticks at it and check what must hold.

    The test keeps its own tally of failures, built only from what it did, and the breaker's
    count has to agree with it exactly.
    """
    rng = random.Random(2026)
    for round_number in range(80):
        threshold = rng.randint(1, 4)
        cooldown = rng.choice([0.0, 1.0, 5.0])
        probes = rng.randint(1, 3)
        epoch = 0  # advances whenever the breaker starts a new generation
        edges: list[tuple[CircuitState, CircuitState]] = []

        def observe(
            _: str,
            old: CircuitState,
            new: CircuitState,
            edges: list[tuple[CircuitState, CircuitState]] = edges,
        ) -> None:
            nonlocal epoch
            edges.append((old, new))
            epoch += 1

        breaker, clock = make(
            threshold=threshold, cooldown=cooldown, probes=probes, on_transition=observe
        )
        held: list[tuple[Permit, int]] = []  # (permit, the epoch it was issued in)
        expected_failures = 0

        for _ in range(400):
            breaker.state  # noqa: B018 - settle any pending open -> half-open move first
            operation = rng.choice(
                ["acquire", "acquire", "success", "failure", "release", "tick", "reset"]
            )
            state_before = breaker.state
            if operation == "acquire":
                permit = breaker.try_acquire()
                if state_before is CLOSED:
                    assert permit is not None, round_number  # closed always admits
                if permit is not None:
                    held.append((permit, epoch))
            elif operation == "tick":
                clock.advance(rng.choice([0.0, 0.5, 1.0, 5.0]))
            elif operation == "reset":
                breaker.reset()
                if state_before is CLOSED:
                    epoch += 1  # a reset of a closed breaker starts a new generation silently
                expected_failures = 0
            elif held:
                permit, issued_in = held.pop(rng.randrange(len(held)))
                current = issued_in == epoch
                if operation == "failure":
                    permit.failure()
                    if current and state_before is CLOSED:
                        expected_failures += 1
                elif operation == "success":
                    permit.success()
                    if current and state_before is CLOSED:
                        expected_failures = 0
                else:
                    permit.release()
                if breaker.state is not CLOSED:
                    expected_failures = 0  # opening forgets the count

            # --- what must always hold ---
            state = breaker.state
            assert breaker.probes_in_flight <= probes, round_number
            if state is CLOSED:
                assert breaker.consecutive_failures == expected_failures, round_number
                assert breaker.consecutive_failures < threshold, round_number
            if state is OPEN and cooldown > 0:
                assert breaker.try_acquire() is None, round_number  # open admits nothing
            assert breaker.remaining_cooldown() >= 0

        assert all(edge in ALLOWED_EDGES for edge in edges), (round_number, edges)
