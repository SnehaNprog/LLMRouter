import logging
import random

import pytest

from app.resilience.breaker import CircuitBreaker, CircuitState
from app.resilience.window import RollingCounter

CLOSED, OPEN, HALF_OPEN = CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def counter(window: float = 10.0, bucket: float = 1.0) -> tuple[RollingCounter, Clock]:
    clock = Clock()
    return RollingCounter(window, bucket_seconds=bucket, clock=clock), clock


def rate_breaker(
    rate: float = 0.5,
    min_requests: int = 10,
    window: float = 60.0,
    threshold: int | None = None,
    cooldown: float = 10.0,
) -> tuple[CircuitBreaker, Clock]:
    clock = Clock()
    breaker = CircuitBreaker(
        "test",
        failure_threshold=threshold,
        failure_rate_threshold=rate,
        min_requests=min_requests,
        window_seconds=window,
        cooldown=cooldown,
        clock=clock,
    )
    return breaker, clock


def call(breaker: CircuitBreaker, ok: bool) -> None:
    permit = breaker.try_acquire()
    assert permit is not None
    permit.success() if ok else permit.failure()


def run(breaker: CircuitBreaker, pattern: str) -> None:
    """'s' is a success and 'f' a failure."""
    for char in pattern:
        call(breaker, char == "s")


# ---------- RollingCounter ----------


def test_an_empty_counter_has_a_zero_rate() -> None:
    c, _ = counter()
    assert (c.total, c.failures, c.rate) == (0, 0, 0.0)


def test_it_counts_calls_and_failures() -> None:
    c, _ = counter()
    for failed in (True, False, False, True, False):
        c.record(failed)
    assert (c.total, c.failures) == (5, 2)
    assert c.rate == pytest.approx(0.4)


def test_events_leave_the_window_once_it_has_passed() -> None:
    c, clock = counter(window=10, bucket=1)
    c.record(True)  # at t0, in bucket 0
    clock.advance(9)
    assert c.total == 1  # the last second the bucket still counts
    clock.advance(1)
    assert (c.total, c.failures) == (0, 0)


def test_only_the_expired_part_leaves() -> None:
    c, clock = counter(window=10, bucket=1)
    c.record(True)
    clock.advance(5)
    c.record(False)
    clock.advance(5)  # the first event is now 10 s old, the second 5 s
    assert (c.total, c.failures) == (1, 0)


def test_a_long_silence_empties_it() -> None:
    c, clock = counter()
    for _ in range(50):
        c.record(True)
    clock.advance(10_000)
    assert (c.total, c.failures, c.rate) == (0, 0, 0.0)
    assert c.buckets_in_use == 0


def test_memory_stays_bounded_however_many_events_there_are() -> None:
    c, clock = counter(window=10, bucket=1)
    for _ in range(5_000):
        c.record(random.random() < 0.5)
        clock.advance(0.01)
    assert c.buckets_in_use <= 10
    assert c.total <= 1_000  # 100 events per second for at most 10 seconds


def test_clear_forgets_everything() -> None:
    c, _ = counter()
    c.record(True)
    c.clear()
    assert (c.total, c.failures, c.buckets_in_use) == (0, 0, 0)


def test_the_counter_matches_a_naive_count_over_random_traffic() -> None:
    rng = random.Random(7)
    c, clock = counter(window=10, bucket=1)
    events: list[tuple[int, bool]] = []  # (bucket index, failed)
    for _ in range(3_000):
        clock.advance(rng.choice([0.0, 0.1, 0.4, 1.0, 3.0]))
        failed = rng.random() < 0.3
        c.record(failed)
        events.append((int(clock.now // 1), failed))
        now = int(clock.now // 1)
        live = [f for index, f in events if index > now - 10]
        assert c.total == len(live)
        assert c.failures == sum(live)


@pytest.mark.parametrize(("window", "bucket"), [(0, 1), (-5, 1), (10, 0), (10, -1), (10, 11)])
def test_the_counter_rejects_nonsense_sizes(window: float, bucket: float) -> None:
    with pytest.raises(ValueError):
        RollingCounter(window, bucket_seconds=bucket)


# ---------- tripping on the rate ----------


def test_it_opens_when_enough_calls_have_failed_often_enough() -> None:
    breaker, _ = rate_breaker(rate=0.5, min_requests=10)
    run(breaker, "sfsfsfsfs")  # 9 calls, 4 failed
    assert breaker.state is CLOSED
    run(breaker, "f")  # 10 calls, 5 failed: 50%
    assert breaker.state is OPEN
    assert breaker.last_trip_reason == "5 of 10 recent calls failed (50%)"


def test_it_does_not_open_below_the_minimum_number_of_calls() -> None:
    breaker, _ = rate_breaker(rate=0.5, min_requests=10)
    run(breaker, "fffffffff")  # nine failures out of nine: 100%, but too few to judge
    assert breaker.state is CLOSED
    assert breaker.failure_rate == 1.0


def test_one_more_call_past_the_minimum_can_open_it() -> None:
    breaker, _ = rate_breaker(rate=0.5, min_requests=10)
    run(breaker, "fffffffff")
    run(breaker, "f")
    assert breaker.state is OPEN


def test_a_rate_below_the_threshold_never_opens_it() -> None:
    breaker, _ = rate_breaker(rate=0.5, min_requests=10)
    run(breaker, "sfs" * 30)  # one failure in three
    assert breaker.state is CLOSED
    assert breaker.failure_rate == pytest.approx(1 / 3)


def test_scattered_failures_open_it_though_they_are_never_consecutive() -> None:
    breaker, _ = rate_breaker(rate=0.5, min_requests=10, threshold=3)
    run(breaker, "sfsfsfsfsf")  # 50%, and never two failures in a row
    assert breaker.state is OPEN
    assert breaker.last_trip_reason == "5 of 10 recent calls failed (50%)"


def test_a_success_never_opens_it_even_if_it_brings_the_count_up_to_the_minimum() -> None:
    breaker, _ = rate_breaker(rate=0.5, min_requests=10)
    run(breaker, "fsfsfsfsf")  # 9 calls, 5 failed: over the rate but under the minimum
    run(breaker, "s")  # 10 calls, 5 failed: a success is no evidence of trouble
    assert breaker.state is CLOSED
    run(breaker, "f")  # the next failure is judged on the full window
    assert breaker.state is OPEN


def test_old_failures_age_out_of_the_rate() -> None:
    breaker, clock = rate_breaker(rate=0.5, min_requests=10, window=60)
    run(breaker, "ffffff")
    clock.advance(61)
    run(breaker, "ssssssssss")
    run(breaker, "f")  # the six old failures no longer count: 1 of 11
    assert breaker.state is CLOSED
    assert breaker.window_requests == 11
    assert breaker.failure_rate == pytest.approx(1 / 11)


def test_a_slow_trickle_never_reaches_the_minimum() -> None:
    breaker, clock = rate_breaker(rate=0.5, min_requests=10, window=60)
    for _ in range(100):
        call(breaker, ok=False)
        clock.advance(30)  # at most two calls are ever in the window
    assert breaker.state is CLOSED


def test_the_rate_alone_can_be_the_only_rule() -> None:
    breaker, _ = rate_breaker(rate=0.8, min_requests=5, threshold=None)
    run(breaker, "ffff")
    assert breaker.state is CLOSED  # four in a row means nothing without the consecutive rule
    run(breaker, "f")
    assert breaker.state is OPEN


def test_consecutive_failures_still_open_it_before_the_minimum_is_reached() -> None:
    breaker, _ = rate_breaker(rate=0.9, min_requests=100, threshold=3)
    run(breaker, "fff")
    assert breaker.state is OPEN
    assert breaker.last_trip_reason == "3 consecutive failures"


def test_with_no_rate_configured_the_window_never_opens_it() -> None:
    clock = Clock()
    breaker = CircuitBreaker("test", failure_threshold=5, clock=clock)
    run(breaker, "ffsffsffsffsffs")
    assert breaker.state is CLOSED


# ---------- the window across state changes ----------


def test_the_window_starts_empty_after_the_breaker_recovers() -> None:
    breaker, clock = rate_breaker(rate=0.5, min_requests=4, cooldown=10)
    run(breaker, "ffff")
    assert breaker.state is OPEN
    assert breaker.window_requests == 0
    clock.advance(10)
    call(breaker, ok=True)  # the trial succeeds
    assert breaker.state is CLOSED
    run(breaker, "fff")  # three failures: below the minimum, and nothing carried over
    assert breaker.state is CLOSED


def test_trial_calls_do_not_feed_the_window() -> None:
    breaker, clock = rate_breaker(rate=0.5, min_requests=4, cooldown=10)
    run(breaker, "ffff")
    clock.advance(10)
    assert breaker.state is HALF_OPEN
    call(breaker, ok=False)  # the trial fails: open again
    assert breaker.state is OPEN
    assert breaker.last_trip_reason == "a trial call failed"
    assert breaker.window_requests == 0


def test_reset_clears_the_window() -> None:
    breaker, _ = rate_breaker(rate=0.5, min_requests=10)
    run(breaker, "ffff")
    breaker.reset()
    assert breaker.window_requests == 0
    assert breaker.failure_rate == 0.0


def test_a_straggler_from_before_a_reset_does_not_enter_the_window() -> None:
    breaker, _ = rate_breaker(rate=0.5, min_requests=10)
    straggler = breaker.acquire()
    breaker.reset()
    straggler.failure()
    assert breaker.window_requests == 0


def test_released_permits_are_not_counted() -> None:
    breaker, _ = rate_breaker(rate=0.5, min_requests=2)
    for _ in range(10):
        breaker.acquire().release()
    assert breaker.window_requests == 0
    assert breaker.state is CLOSED


def test_the_trip_reason_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    breaker, _ = rate_breaker(rate=0.5, min_requests=4)
    with caplog.at_level(logging.INFO, logger="app.resilience.breaker"):
        run(breaker, "ffff")
    record = next(r for r in caplog.records if r.__dict__.get("to_state") == "open")
    assert record.__dict__["reason"] == "4 of 4 recent calls failed (100%)"


# ---------- validation ----------


@pytest.mark.parametrize("rate", [0, -0.1, 1.5])
def test_the_rate_must_be_above_zero_and_at_most_one(rate: float) -> None:
    with pytest.raises(ValueError, match="failure_rate_threshold"):
        CircuitBreaker("x", failure_rate_threshold=rate)


def test_a_rate_of_exactly_one_is_allowed() -> None:
    CircuitBreaker("x", failure_rate_threshold=1.0)


def test_min_requests_must_be_positive() -> None:
    with pytest.raises(ValueError, match="min_requests"):
        CircuitBreaker("x", failure_rate_threshold=0.5, min_requests=0)


def test_at_least_one_rule_is_required() -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreaker("x", failure_threshold=None)


def test_a_zero_failure_threshold_is_still_rejected() -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreaker("x", failure_threshold=0)


def test_the_window_must_be_positive() -> None:
    with pytest.raises(ValueError, match="window_seconds"):
        CircuitBreaker("x", failure_rate_threshold=0.5, window_seconds=0)


# ---------- random traffic ----------


def test_it_opens_exactly_when_a_naive_model_says_it_should() -> None:
    rng = random.Random(11)
    for _ in range(40):
        rate = rng.choice([0.3, 0.5, 0.75, 1.0])
        minimum = rng.choice([1, 3, 8])
        breaker, clock = rate_breaker(rate=rate, min_requests=minimum, window=10, cooldown=5)
        history: list[tuple[int, bool]] = []  # (bucket, failed) since the breaker last closed
        for _ in range(300):
            clock.advance(rng.choice([0.0, 0.2, 1.0, 4.0]))
            if breaker.state is not CLOSED:
                history.clear()
                clock.advance(5)
                if rng.random() < 0.5:
                    call(breaker, ok=True)
                    continue
                breaker.reset()
                continue
            failed = rng.random() < 0.6
            call(breaker, ok=not failed)
            now = int(clock.now // 1)
            history.append((now, failed))
            live = [f for index, f in history if index > now - 10]
            # only a failure puts the window on trial
            should_open = failed and len(live) >= minimum and sum(live) / len(live) >= rate
            assert (breaker.state is OPEN) == should_open
