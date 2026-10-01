import math
import random
from collections.abc import Callable

from pydantic import BaseModel


class Interval(BaseModel):
    low: float
    high: float

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


def bootstrap_ci(
    n: int,
    statistic: Callable[[list[int]], float | None],
    *,
    resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval | None:
    """A percentile bootstrap confidence interval for a statistic of ``n`` items.

    The items are resampled with replacement ``resamples`` times, ``statistic`` is worked out on
    each resample (it receives the chosen item indices), and the middle ``level`` of the results
    is the interval. It shows how much the answer could move if the benchmark were a different
    draw of similar prompts, which a point estimate on 90 prompts hides. A resample for which the
    statistic is undefined (None) is skipped; if none is defined there is no interval.

    The seed fixes the draws, so the same data always gives the same interval.
    """
    if n < 1:
        raise ValueError("there is nothing to resample")
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    rng = random.Random(seed)
    population = range(n)
    values = []
    for _ in range(resamples):
        value = statistic(rng.choices(population, k=n))
        if value is not None:
            values.append(value)
    if not values:
        return None
    values.sort()
    # Ranks, nudged by a hair so that float error (1 - 0.90) / 2 = 0.04999... cannot move one.
    tail = (1 - level) / 2
    low_rank = math.floor(tail * len(values) + 1e-9)
    high_rank = math.ceil((1 - tail) * len(values) - 1e-9) - 1
    low = values[max(0, min(len(values) - 1, low_rank))]
    high = values[max(0, min(len(values) - 1, high_rank))]
    return Interval(low=low, high=high)
