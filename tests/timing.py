"""Helpers for asserting on quantities that change as time passes.

A token bucket keeps refilling while a test runs, so "the balance is exactly X" is never quite
true, and "X plus a little refill" is only true if "a little" is worked out from how long the test
really took, not guessed. A guess that holds on a fast laptop fails on a busy CI runner; these
helpers hold on any machine (and `make test-slow` checks that they do).
"""

import time


def refill_since(started: float, limit_per_minute: int) -> float:
    """Tokens a bucket of ``limit_per_minute`` has regained since ``started`` (time.monotonic())."""
    return limit_per_minute / 60 * (time.monotonic() - started)


def balance_range(
    spent: int, started: float, limit: int = 1_000, below: int = 10
) -> tuple[float, float]:
    """Where a full bucket should stand after ``spent`` tokens were really used.

    The lowest it can be is what is left (give or take ``below``); the highest is what is left plus
    everything it has refilled since ``started``, and never more than the bucket holds.
    """
    return limit - spent - below, min(limit, limit - spent + refill_since(started, limit) + 5)
