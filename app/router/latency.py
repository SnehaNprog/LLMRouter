from dataclasses import dataclass


@dataclass(frozen=True)
class LatencyStats:
    ewma: float  # smoothed latency in seconds
    last: float
    samples: int


class LatencyTracker:
    """Exponentially weighted moving average of latency, per deployment.

    Each new observation moves the average by ``alpha`` of the way towards it, so recent
    behaviour dominates and a single slow request cannot swing the estimate far. Only
    successful calls are recorded: a fast failure would make a broken deployment look quick.
    """

    def __init__(self, alpha: float = 0.2) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be greater than 0 and at most 1")
        self.alpha = alpha
        self._stats: dict[str, LatencyStats] = {}

    def observe(self, deployment_id: str, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("latency must not be negative")
        previous = self._stats.get(deployment_id)
        if previous is None:
            self._stats[deployment_id] = LatencyStats(ewma=seconds, last=seconds, samples=1)
            return
        ewma = self.alpha * seconds + (1 - self.alpha) * previous.ewma
        self._stats[deployment_id] = LatencyStats(
            ewma=ewma, last=seconds, samples=previous.samples + 1
        )

    def ewma(self, deployment_id: str) -> float | None:
        stats = self._stats.get(deployment_id)
        return None if stats is None else stats.ewma

    def stats(self, deployment_id: str) -> LatencyStats | None:
        return self._stats.get(deployment_id)

    def snapshot(self) -> dict[str, LatencyStats]:
        return dict(self._stats)
