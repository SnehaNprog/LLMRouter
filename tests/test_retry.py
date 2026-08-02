import random

import pytest

from app.providers.errors import (
    AuthFailed,
    BadRequest,
    ProviderDown,
    ProviderError,
    RateLimited,
    Timeout,
)
from app.router.retry import Decision, RetryPolicy, backoff_delay, classify

# ---------- classification ----------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (Timeout("x"), Decision.RETRY),
        (ProviderDown("x"), Decision.RETRY),
        (RateLimited("x"), Decision.SWITCH),
        (AuthFailed("x"), Decision.SWITCH),
        (BadRequest("x"), Decision.FAIL),
        (ProviderError("x"), Decision.FAIL),
        (ValueError("bug"), Decision.FAIL),
        (KeyError("bug"), Decision.FAIL),
    ],
)
def test_classification_table(error: BaseException, expected: Decision) -> None:
    assert classify(error) is expected


def test_unknown_retryable_provider_error_follows_its_flag() -> None:
    class FlakyError(ProviderError):
        retryable = True

    assert classify(FlakyError("x")) is Decision.RETRY


# ---------- backoff ----------


class TopOfRange(random.Random):
    """Always draws the upper bound, which exposes the deterministic ceiling."""

    def uniform(self, a: float, b: float) -> float:
        return b


class BottomOfRange(random.Random):
    def uniform(self, a: float, b: float) -> float:
        return a


POLICY = RetryPolicy(max_attempts=5, base_delay=1.0, multiplier=2.0, max_delay=10.0)


def test_ceiling_grows_exponentially() -> None:
    delays = [backoff_delay(n, POLICY, TopOfRange()) for n in range(1, 5)]

    assert delays == [1.0, 2.0, 4.0, 8.0]


def test_ceiling_is_capped_at_max_delay() -> None:
    assert backoff_delay(5, POLICY, TopOfRange()) == 10.0
    assert backoff_delay(50, POLICY, TopOfRange()) == 10.0


def test_jitter_can_shrink_the_delay_to_zero() -> None:
    assert backoff_delay(3, POLICY, BottomOfRange()) == 0.0


def test_jittered_delays_stay_within_bounds() -> None:
    rng = random.Random(123)

    for retry in range(1, 8):
        ceiling = min(10.0, 1.0 * 2 ** (retry - 1))
        for _ in range(200):
            assert 0.0 <= backoff_delay(retry, POLICY, rng) <= ceiling


def test_jitter_actually_varies() -> None:
    rng = random.Random(9)

    delays = {backoff_delay(3, POLICY, rng) for _ in range(50)}

    assert len(delays) > 40


def test_same_seed_gives_same_delays() -> None:
    first = [backoff_delay(n, POLICY, random.Random(5)) for n in (1, 2, 3)]
    second = [backoff_delay(n, POLICY, random.Random(5)) for n in (1, 2, 3)]

    assert first == second


def test_retry_after_is_a_floor() -> None:
    assert backoff_delay(1, POLICY, BottomOfRange(), retry_after=4.0) == 4.0


def test_retry_after_is_bounded_by_max_delay() -> None:
    assert backoff_delay(1, POLICY, BottomOfRange(), retry_after=600.0) == 10.0


def test_retry_after_never_shortens_a_longer_backoff() -> None:
    assert backoff_delay(4, POLICY, TopOfRange(), retry_after=1.0) == 8.0


def test_zero_base_delay_means_no_waiting() -> None:
    policy = RetryPolicy(base_delay=0.0)

    assert backoff_delay(3, policy, random.Random(1)) == 0.0


def test_retry_numbers_start_at_one() -> None:
    with pytest.raises(ValueError, match="starts at 1"):
        backoff_delay(0, POLICY, random.Random(0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"base_delay": -1.0},
        {"max_delay": -1.0},
        {"multiplier": 0.5},
    ],
)
def test_invalid_policies_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)  # type: ignore[arg-type]


def test_default_policy() -> None:
    policy = RetryPolicy()

    assert (policy.max_attempts, policy.base_delay, policy.multiplier, policy.max_delay) == (
        3,
        0.5,
        2.0,
        8.0,
    )
