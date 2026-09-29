import random
from typing import Any

import pytest

from app.core.config import Settings
from app.providers.errors import ProviderDown
from app.providers.fake import FakeProvider
from app.resilience.board import BreakerBoard, BreakerSettings, DeploymentUnavailable
from app.resilience.breaker import CircuitBreaker, CircuitState, Permit
from app.router.deployment import Deployment
from tests.conftest import model_config

CLOSED, OPEN, HALF_OPEN = CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def half_open(
    calls: int = 1, successes: int = 1, timeout: float | None = 30.0, cooldown: float = 10.0
) -> tuple[CircuitBreaker, Clock]:
    clock = Clock()
    breaker = CircuitBreaker(
        "test",
        failure_threshold=1,
        cooldown=cooldown,
        half_open_max_calls=calls,
        half_open_successes=successes,
        probe_timeout=timeout,
        clock=clock,
    )
    breaker.acquire().failure()
    clock.advance(cooldown)
    assert breaker.state is HALF_OPEN
    return breaker, clock


def take(breaker: CircuitBreaker) -> Permit:
    permit = breaker.try_acquire()
    assert permit is not None
    return permit


# ---------- several successes to close ----------


def test_it_needs_the_configured_number_of_successful_trials() -> None:
    breaker, _ = half_open(successes=3)
    for expected in (HALF_OPEN, HALF_OPEN, CLOSED):
        take(breaker).success()
        assert breaker.state is expected


def test_one_success_is_the_default() -> None:
    breaker, _ = half_open()
    take(breaker).success()
    assert breaker.state is CLOSED


def test_a_failed_trial_reopens_it_whatever_came_before() -> None:
    breaker, _ = half_open(successes=3)
    take(breaker).success()
    take(breaker).success()
    take(breaker).failure()
    assert breaker.state is OPEN


def test_progress_does_not_carry_over_to_the_next_half_open_period() -> None:
    breaker, clock = half_open(successes=2, cooldown=10)
    take(breaker).success()
    take(breaker).failure()  # reopened with one success banked
    clock.advance(10)
    take(breaker).success()
    assert breaker.state is HALF_OPEN  # one success, not two
    take(breaker).success()
    assert breaker.state is CLOSED


def test_a_finished_trial_frees_its_slot_for_the_next() -> None:
    breaker, _ = half_open(calls=1, successes=2)
    take(breaker).success()
    assert breaker.try_acquire() is not None  # the slot is free again after a success


def test_trials_are_capped_at_the_same_time() -> None:
    breaker, _ = half_open(calls=2, successes=2)
    first, second = take(breaker), take(breaker)
    assert breaker.try_acquire() is None
    assert breaker.probes_in_flight == 2
    first.success()
    assert breaker.probes_in_flight == 1
    second.success()
    assert breaker.state is CLOSED


def test_a_released_trial_frees_its_slot_without_counting() -> None:
    breaker, _ = half_open(successes=2)
    take(breaker).release()
    assert breaker.probes_in_flight == 0
    take(breaker).success()
    assert breaker.state is HALF_OPEN  # the release was not a success


def test_reset_forgets_progress() -> None:
    breaker, _ = half_open(successes=2)
    take(breaker).success()
    breaker.reset()
    assert breaker.state is CLOSED
    breaker.acquire().failure()
    assert breaker.state is OPEN


def test_a_straggler_from_an_earlier_period_does_not_count_as_a_trial() -> None:
    breaker, clock = half_open(successes=2, cooldown=10)
    old = take(breaker)
    assert breaker.try_acquire() is None  # the only slot is held by `old`
    clock.advance(30)  # `old` is reaped
    fresh = take(breaker)
    old.success()
    assert breaker.state is HALF_OPEN
    fresh.success()
    assert breaker.state is HALF_OPEN  # one real success of two


# ---------- trials that are never reported ----------


def test_a_lost_trial_frees_its_slot_after_the_timeout() -> None:
    breaker, clock = half_open(timeout=30)
    take(breaker)  # never reported
    clock.advance(29)
    assert breaker.try_acquire() is None
    clock.advance(1)
    assert breaker.try_acquire() is not None


def test_the_report_of_a_reclaimed_trial_is_ignored() -> None:
    breaker, clock = half_open(timeout=30)
    lost = take(breaker)
    clock.advance(30)
    current = take(breaker)

    lost.failure()  # a late failure must not reopen the breaker
    assert breaker.state is HALF_OPEN
    assert breaker.probes_in_flight == 1  # the live trial still holds the slot
    current.success()
    assert breaker.state is CLOSED


def test_a_late_success_from_a_reclaimed_trial_does_not_close_it() -> None:
    breaker, clock = half_open(timeout=30)
    lost = take(breaker)
    clock.advance(30)
    lost.success()
    assert breaker.state is HALF_OPEN


def test_probes_in_flight_does_not_count_lost_trials() -> None:
    breaker, clock = half_open(calls=2, timeout=30)
    take(breaker)
    take(breaker)
    clock.advance(30)
    assert breaker.probes_in_flight == 0


def test_without_a_timeout_a_lost_trial_holds_its_slot() -> None:
    breaker, clock = half_open(timeout=None)
    take(breaker)
    clock.advance(1_000_000)
    assert breaker.try_acquire() is None


# ---------- validation ----------


def test_successes_must_be_positive() -> None:
    with pytest.raises(ValueError, match="half_open_successes"):
        CircuitBreaker("x", half_open_successes=0)


@pytest.mark.parametrize("timeout", [0, -1])
def test_the_probe_timeout_must_be_positive(timeout: float) -> None:
    with pytest.raises(ValueError, match="probe_timeout"):
        CircuitBreaker("x", probe_timeout=timeout)


# ---------- random traffic ----------


def test_trials_in_flight_never_exceed_the_limit_and_it_always_ends_up_closed() -> None:
    rng = random.Random(5)
    for _ in range(60):
        calls, successes = rng.randint(1, 4), rng.randint(1, 3)
        breaker, clock = half_open(calls=calls, successes=successes, timeout=20, cooldown=5)
        held: list[Permit] = []
        for _ in range(200):
            clock.advance(rng.choice([0.0, 0.5, 5.0, 25.0]))
            action = rng.random()
            if action < 0.5:
                permit = breaker.try_acquire()
                if permit is not None:
                    held.append(permit)
            elif held:
                permit = held.pop(rng.randrange(len(held)))
                rng.choice([permit.success, permit.success, permit.failure, permit.release])()
            assert breaker.probes_in_flight <= calls
        # a healthy dependency: everything now succeeds, and the breaker recovers
        for _ in range(10):
            clock.advance(60)
            permit = breaker.try_acquire()
            if permit is not None:
                permit.success()
        assert breaker.state is CLOSED


# ---------- through the board and the settings ----------


def deployment() -> Deployment:
    return Deployment.create(model_config(), FakeProvider())


def test_the_board_passes_the_probe_settings_to_both_breakers() -> None:
    clock = Clock()
    board = BreakerBoard(
        BreakerSettings(
            failure_threshold=1,
            failure_rate_threshold=None,
            cooldown=10,
            half_open_successes=2,
            probe_timeout=5,
        ),
        clock=clock,
    )
    target = deployment()
    board.acquire(target).failure(ProviderDown("x"))
    clock.advance(10)

    board.acquire(target).success()
    assert board.is_open(target) is False
    assert board.deployment_breaker(target).state is HALF_OPEN  # one of two
    assert board.provider_breaker(target).state is HALF_OPEN
    board.acquire(target).success()
    assert board.deployment_breaker(target).state is CLOSED
    assert board.provider_breaker(target).state is CLOSED


def test_a_lost_trial_through_the_board_is_reclaimed() -> None:
    clock = Clock()
    board = BreakerBoard(
        BreakerSettings(
            failure_threshold=1, failure_rate_threshold=None, cooldown=10, probe_timeout=5
        ),
        clock=clock,
    )
    target = deployment()
    board.acquire(target).failure(ProviderDown("x"))
    clock.advance(10)
    board.acquire(target)  # never reported
    with pytest.raises(DeploymentUnavailable):
        board.acquire(target)
    clock.advance(5)
    board.acquire(target).success()


@pytest.mark.parametrize(
    "field", ["breaker_half_open_calls", "breaker_half_open_successes", "breaker_probe_timeout"]
)
def test_nonsense_settings_are_rejected(field: str) -> None:
    values: dict[str, Any] = {field: 0}
    with pytest.raises(ValueError):
        Settings(_env_file=None, **values)
