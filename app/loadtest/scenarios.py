import asyncio
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Literal

import httpx
from pydantic import BaseModel

from app.loadtest.harness import Stack
from app.loadtest.load import (
    LoadSummary,
    Outcome,
    Run,
    Sample,
    chat_sender,
    closed_loop,
    open_loop,
    summarize,
)


@dataclass(frozen=True)
class Spec:
    """One measured run: what to send, to whom, and how hard."""

    name: str
    title: str
    model: str = "small"
    target: Literal["gateway", "stub"] = "gateway"
    stream: bool = False
    mode: Literal["closed", "open"] = "closed"
    concurrency: int = 50  # closed loop
    rate: float = 100.0  # open loop, requests per second
    duration: float = 10.0
    warmup: float = 2.0


class ScenarioResult(BaseModel):
    name: str
    title: str
    summary: LoadSummary | None = None
    details: dict[str, Any] = {}
    gateway_cpu_percent: float | None = None  # of one core, over the measured seconds
    gateway_rss_mb: float | None = None
    upstream_requests: int | None = None  # how many requests the stub actually received


@dataclass(frozen=True)
class Job:
    """Everything one load-generating process needs: plain data, so it can cross a process."""

    base_url: str
    api_key: str | None
    model: str
    stream: bool
    mode: Literal["closed", "open"]
    concurrency: int
    rate: float
    duration: float
    warmup: float
    start_at: float = 0.0  # wall-clock time (time.time()) at which every process begins


def split(job: Job, parts: int, start_at: float) -> list[Job]:
    """Share one job between ``parts`` processes: the connections or the arrival rate divide
    between them, and all begin at the same moment."""
    if parts < 1:
        raise ValueError("parts must be at least 1")
    base, extra = divmod(job.concurrency, parts)
    jobs = []
    for n in range(parts):
        share = base + (1 if n < extra else 0)
        if job.mode == "closed" and share == 0:
            continue  # more processes than connections: the surplus has nothing to do
        jobs.append(replace(job, concurrency=share, rate=job.rate / parts, start_at=start_at))
    return jobs


async def execute(job: Job) -> Run:
    """Run one job in this process."""
    connections = job.concurrency if job.mode == "closed" else int(job.rate * 2) + 50
    delay = job.start_at - time.time()
    if delay > 0:
        await asyncio.sleep(delay)
    async with httpx.AsyncClient(
        base_url=job.base_url,
        timeout=httpx.Timeout(60.0),
        limits=httpx.Limits(
            max_connections=connections + 20, max_keepalive_connections=connections + 20
        ),
    ) as client:
        send = chat_sender(client, model=job.model, api_key=job.api_key, stream=job.stream)
        if job.mode == "closed":
            return await closed_loop(
                send, concurrency=job.concurrency, duration=job.duration, warmup=job.warmup
            )
        return await open_loop(send, rate=job.rate, duration=job.duration, warmup=job.warmup)


Packed = list[tuple[float, float, int, str, float | None, float | None]]


def pack(run: Run) -> Packed:
    return [
        (s.started, s.latency, s.outcome.status, s.outcome.kind, s.outcome.ttfb, s.behind)
        for s in run.samples
    ]


def unpack(packed: Packed, wall: float) -> Run:
    return Run(
        [
            Sample(st, lat, Outcome(status, kind, ttfb), behind)
            for st, lat, status, kind, ttfb, behind in packed
        ],
        wall=wall,
    )


def _worker(job: Job) -> Packed:
    """A process-pool entry point: run the job and send back plain tuples."""
    return pack(asyncio.run(execute(job)))


def merge(runs: list[Run], wall: float) -> Run:
    return Run([sample for run in runs for sample in run.samples], wall=wall)


async def generate(job: Job, generators: int) -> Run:
    """Produce the load, from one process or from several.

    A single Python process running an HTTP client saturates one core at somewhere around 600
    requests a second and then measures itself, not the thing under test. Beyond that, the load is
    shared out between processes that start together and whose samples are merged.
    """
    if generators <= 1:
        return await execute(job)
    jobs = split(job, generators, start_at=time.time() + 1.5)
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        packed = await asyncio.gather(*(loop.run_in_executor(pool, _worker, j) for j in jobs))
    return merge([unpack(p, job.duration) for p in packed], job.duration)


async def run_spec(
    stack: Stack, spec: Spec, *, key: str | None = None, generators: int = 1
) -> ScenarioResult:
    """Run one scenario against the stack and report what the load generator saw, together with
    what the gateway process spent and what the upstream actually received."""
    key = key if spec.target == "gateway" else None
    base = stack.gateway_url if spec.target == "gateway" else stack.stub_url
    job = Job(
        base_url=base,
        api_key=key,
        model=spec.model,
        stream=spec.stream,
        mode=spec.mode,
        concurrency=spec.concurrency,
        rate=spec.rate,
        duration=spec.duration,
        warmup=spec.warmup,
    )
    stack.stub_reset()
    before = stack.gateway_process_stats()
    run = await generate(job, generators)
    after = stack.gateway_process_stats()
    result = ScenarioResult(
        name=spec.name,
        title=spec.title,
        summary=summarize(run),
        upstream_requests=stack.stub_stats()["requests"],
    )
    if before is not None and after is not None:
        seconds = spec.duration + spec.warmup
        result.gateway_cpu_percent = round(
            100 * (after["cpu_seconds"] - before["cpu_seconds"]) / seconds, 1
        )
        result.gateway_rss_mb = after["rss_mb"]
    return result


async def prewarm(
    stack: Stack, key: str, *, concurrency: int, generators: int = 1, seconds: float = 3.0
) -> None:
    """Send some unrecorded traffic everywhere first, so no scenario pays for being the first.

    Without it the first scenario measured absorbs cold starts (lazy imports, connection set-up,
    caches) and looks worse than the ones after it, which is not what it costs in steady state.
    """
    for target, api_key in (("stub", None), ("gateway", key)):
        spec = Spec(
            "warmup",
            "warm-up",
            target="stub" if target == "stub" else "gateway",
            concurrency=concurrency,
            duration=seconds,
            warmup=0.0,
        )
        await run_spec(stack, spec, key=api_key, generators=generators)


async def run_outage(
    stack: Stack, key: str, *, rate: float, phase_seconds: float
) -> ScenarioResult:
    """The upstream goes completely down for a while and then comes back.

    Requests keep arriving at ``rate`` per second throughout. What matters is how many of them
    still reached the dead upstream once the circuit breaker had opened (it should be very few),
    and how quickly good answers resume once the upstream is back.
    """
    stack.stub_reset()
    phases: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        base_url=stack.gateway_url, timeout=60.0, limits=httpx.Limits(max_connections=2000)
    ) as client:
        send = chat_sender(client, model="small", api_key=key)
        for name, error_rate in (("healthy", 0.0), ("outage", 1.0), ("recovery", 0.0)):
            stack.stub_configure(error_rate=error_rate)
            seen_before = stack.stub_stats()["requests"]
            run = await open_loop(send, rate=rate, duration=phase_seconds)
            summary = summarize(run)
            phases.append(
                {
                    "phase": name,
                    "sent": summary.requests,
                    "ok": summary.ok,
                    "errors": summary.errors,
                    "reached_upstream": stack.stub_stats()["requests"] - seen_before,
                    "p95_ms": summary.latency_ms.p95,
                }
            )
    stack.stub_configure(error_rate=0.0)
    return ScenarioResult(
        name="outage",
        title=f"Upstream down for {phase_seconds:.0f}s at {rate:.0f} req/s",
        details={"phases": phases},
    )


async def run_rate_limit(
    stack: Stack, *, limit: int, requests: int, concurrency: int
) -> ScenarioResult:
    """A key limited to ``limit`` requests a minute is hit by ``requests`` at once.

    Exactly ``limit`` of them should be let through, however many arrive at the same instant:
    the limit is a promise, and concurrency is where a limiter that reads then writes breaks it.
    """
    key = stack.make_key(rpm_limit=limit, name="rate limited")
    stack.stub_reset()
    outcomes: list[Outcome] = []
    gate = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=stack.gateway_url, timeout=60.0) as client:
        send = chat_sender(client, model="small", api_key=key)

        async def one() -> None:
            async with gate:
                outcomes.append(await send())

        await asyncio.gather(*(one() for _ in range(requests)))
    allowed = sum(1 for o in outcomes if o.status == 200)
    rejected = sum(1 for o in outcomes if o.status == 429)
    return ScenarioResult(
        name="rate_limit",
        title=f"{requests} requests at once against a {limit}/minute key",
        details={
            "limit": limit,
            "sent": requests,
            "allowed": allowed,
            "rejected_429": rejected,
            "other": requests - allowed - rejected,
            "exact": allowed == limit,
            "reached_upstream": stack.stub_stats()["requests"],
        },
    )


def default_specs(*, concurrency: int, duration: float, warmup: float) -> dict[str, Spec]:
    def spec(name: str, title: str, **changes: Any) -> Spec:
        return Spec(
            name, title, concurrency=concurrency, duration=duration, warmup=warmup, **changes
        )

    return {
        "direct": spec("direct", "Straight to the stub (the ceiling of this setup)", target="stub"),
        "gateway": spec("gateway", "Through the gateway, named model"),
        "auto": spec("auto", "Through the gateway, model=auto", model="auto"),
        "stream": spec("stream", "Through the gateway, streamed", stream=True),
    }


def sweep_specs(levels: list[int], *, duration: float, warmup: float) -> list[Spec]:
    return [
        Spec(f"gateway@{n}", f"Gateway at {n} concurrent", concurrency=n, duration=duration,
             warmup=warmup)
        for n in levels
    ]  # fmt: skip


def paced_specs(rates: list[float], *, duration: float, warmup: float) -> list[Spec]:
    return [
        Spec(f"open@{int(r)}", f"Gateway at a steady {int(r)} req/s", mode="open", rate=r,
             duration=duration, warmup=warmup)
        for r in rates
    ]  # fmt: skip
