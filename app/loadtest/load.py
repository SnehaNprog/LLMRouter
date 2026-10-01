import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel

from app.evaluation.runner import percentile


@dataclass
class Outcome:
    """What one request came to."""

    status: int  # the HTTP status, or 0 if no response arrived
    kind: str  # "ok", "http_503", "timeout", "connect", "truncated"...
    ttfb: float | None = None  # seconds to the first byte of a streamed body

    @property
    def ok(self) -> bool:
        return self.kind == "ok"


@dataclass
class Sample:
    started: float  # seconds since the run began
    latency: float  # the request's own duration, seconds
    outcome: Outcome
    behind: float | None = None  # open loop: how long after its scheduled moment it began


@dataclass
class Run:
    samples: list[Sample] = field(default_factory=list)
    wall: float = 0.0  # seconds the measured part of the run lasted


Send = Callable[[], Awaitable[Outcome]]


class Percentiles(BaseModel):
    p50: float | None
    p90: float | None
    p95: float | None
    p99: float | None
    max: float | None
    mean: float | None


class LoadSummary(BaseModel):
    requests: int
    ok: int
    errors: dict[str, int]  # by kind
    seconds: float
    throughput_rps: float  # successful requests per second
    latency_ms: Percentiles  # of the successful ones
    ttfb_ms: Percentiles | None  # streaming only
    behind_ms: Percentiles | None  # open loop only: lateness of the load generator itself

    @property
    def error_rate(self) -> float:
        return 0.0 if self.requests == 0 else (self.requests - self.ok) / self.requests


def _percentiles(values: list[float]) -> Percentiles:
    def ms(fraction: float) -> float | None:
        found = percentile(values, fraction)
        return None if found is None else round(found * 1000, 2)

    return Percentiles(
        p50=ms(0.50),
        p90=ms(0.90),
        p95=ms(0.95),
        p99=ms(0.99),
        max=ms(1.0),
        mean=round(sum(values) / len(values) * 1000, 2) if values else None,
    )


def summarize(run: Run) -> LoadSummary:
    good = [s for s in run.samples if s.outcome.ok]
    errors: dict[str, int] = {}
    for sample in run.samples:
        if not sample.outcome.ok:
            errors[sample.outcome.kind] = errors.get(sample.outcome.kind, 0) + 1
    ttfb = [s.outcome.ttfb for s in good if s.outcome.ttfb is not None]
    late = [s.behind for s in run.samples if s.behind is not None]
    return LoadSummary(
        requests=len(run.samples),
        ok=len(good),
        errors=dict(sorted(errors.items())),
        seconds=round(run.wall, 3),
        throughput_rps=round(len(good) / run.wall, 2) if run.wall > 0 else 0.0,
        latency_ms=_percentiles([s.latency for s in good]),
        ttfb_ms=_percentiles(ttfb) if ttfb else None,
        behind_ms=_percentiles(late) if late else None,
    )


async def _timed(send: Send, origin: float, scheduled: float | None = None) -> Sample:
    began = time.perf_counter()
    try:
        outcome = await send()
    except httpx.TimeoutException:
        outcome = Outcome(0, "timeout")
    except httpx.ConnectError:
        outcome = Outcome(0, "connect")
    except httpx.HTTPError:
        outcome = Outcome(0, "transport")
    latency = time.perf_counter() - began
    behind = None if scheduled is None else max(0.0, began - scheduled)
    return Sample(began - origin, latency, outcome, behind)


async def closed_loop(send: Send, *, concurrency: int, duration: float, warmup: float = 0.0) -> Run:
    """``concurrency`` workers, each sending the next request the moment the last one finishes.

    Throughput is then whatever the system sustains at that much parallelism. Requests that
    *began* during the first ``warmup`` seconds are left out of the figures.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if duration <= 0 or warmup < 0:
        raise ValueError("duration must be positive and warmup not negative")
    origin = time.perf_counter()
    stop_at = origin + warmup + duration
    samples: list[Sample] = []

    async def worker() -> None:
        while time.perf_counter() < stop_at:
            sample = await _timed(send, origin)
            if sample.started >= warmup:
                samples.append(sample)

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return Run(samples, wall=duration)


async def open_loop(
    send: Send, *, rate: float, duration: float, warmup: float = 0.0, max_in_flight: int = 5000
) -> Run:
    """Requests arrive at ``rate`` per second whether or not earlier ones have finished.

    A closed loop slows down when the system does, which hides how bad it is (the "coordinated
    omission" problem). Here arrivals follow the clock, and each sample records how late it was
    started relative to its scheduled moment, so a load generator that cannot keep up shows it.
    """
    if rate <= 0 or duration <= 0 or warmup < 0:
        raise ValueError("rate and duration must be positive and warmup not negative")
    origin = time.perf_counter()
    total = int((warmup + duration) * rate)
    gate = asyncio.Semaphore(max_in_flight)
    tasks: list[asyncio.Task[Sample]] = []

    async def one(scheduled: float) -> Sample:
        async with gate:
            return await _timed(send, origin, scheduled)

    for n in range(total):
        scheduled = origin + n / rate
        pause = scheduled - time.perf_counter()
        if pause > 0:
            await asyncio.sleep(pause)
        tasks.append(asyncio.create_task(one(scheduled)))
    done = await asyncio.gather(*tasks)
    samples = [s for s in done if s.started >= warmup]
    return Run(samples, wall=duration)


def chat_sender(
    client: httpx.AsyncClient,
    *,
    model: str,
    api_key: str | None = None,
    stream: bool = False,
    path: str = "/v1/chat/completions",
) -> Send:
    """A request that asks ``model`` a short question and reports how it went."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "stream": stream,
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def send() -> Outcome:
        if not stream:
            response = await client.post(path, json=body, headers=headers)
            if response.status_code == 200:
                return Outcome(200, "ok")
            return Outcome(response.status_code, f"http_{response.status_code}")
        started = time.perf_counter()
        async with client.stream("POST", path, json=body, headers=headers) as response:
            if response.status_code != 200:
                await response.aread()
                return Outcome(response.status_code, f"http_{response.status_code}")
            first: float | None = None
            finished = False
            async for line in response.aiter_lines():
                if first is None and line.startswith("data:"):
                    first = time.perf_counter() - started
                if line.strip() == "data: [DONE]":
                    finished = True
            return Outcome(200, "ok" if finished else "truncated", ttfb=first)

    return send
