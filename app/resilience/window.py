import math
import time
from collections.abc import Callable


class RollingCounter:
    """Counts calls and failures over the last ``window_seconds``, in constant memory.

    Time is cut into buckets of ``bucket_seconds`` and only the bucket totals are kept, so the
    memory used depends on the length of the window, never on how much traffic there is. The
    price is that the window is accurate to within one bucket: it always covers the current
    bucket plus the ones before it that fit, so its true length is between ``window - bucket``
    and ``window``.

    Running totals are maintained as buckets come and go, so reading them is O(1).
    """

    def __init__(
        self,
        window_seconds: float,
        *,
        bucket_seconds: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if bucket_seconds <= 0 or bucket_seconds > window_seconds:
            raise ValueError("bucket_seconds must be positive and no longer than the window")
        self._bucket_seconds = bucket_seconds
        self._bucket_count = math.ceil(window_seconds / bucket_seconds)
        self._clock = clock
        self._buckets: dict[int, list[int]] = {}  # bucket index -> [calls, failures]
        self._total = 0
        self._failures = 0
        self._newest = 0

    def _index(self) -> int:
        return math.floor(self._clock() / self._bucket_seconds)

    def _expire(self, index: int) -> None:
        if index == self._newest and self._buckets:
            return  # nothing has moved on since the last look
        self._newest = index
        oldest_kept = index - self._bucket_count + 1
        for stale in [i for i in self._buckets if i < oldest_kept]:
            calls, failures = self._buckets.pop(stale)
            self._total -= calls
            self._failures -= failures

    def record(self, failed: bool) -> None:
        index = self._index()
        self._expire(index)
        bucket = self._buckets.setdefault(index, [0, 0])
        bucket[0] += 1
        self._total += 1
        if failed:
            bucket[1] += 1
            self._failures += 1

    @property
    def total(self) -> int:
        self._expire(self._index())
        return self._total

    @property
    def failures(self) -> int:
        self._expire(self._index())
        return self._failures

    @property
    def rate(self) -> float:
        """The share of calls in the window that failed (0 when there have been none)."""
        total = self.total
        return self._failures / total if total else 0.0

    def clear(self) -> None:
        self._buckets.clear()
        self._total = self._failures = 0

    @property
    def buckets_in_use(self) -> int:
        return len(self._buckets)
