import asyncio
import json
import math
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from app.evaluation.dataset import Benchmark, BenchmarkItem
from app.evaluation.pricing import PriceBook

RETRYABLE = {429, 502, 503, 504}
Sleep = Callable[[float], Awaitable[None]]


class EvalRecord(BaseModel):
    """What happened when one benchmark item was sent to the gateway."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    run: str  # the label of the run, e.g. "baseline"
    requested_model: str
    served_model: str | None = None  # the alias that answered (differs from the request for auto)
    ok: bool
    status_code: int | None = None
    error: str | None = None
    reply: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None  # None when the tokens or the model's price are unknown
    latency_s: float | None = None  # of the final attempt, which is the one that answered
    attempts: int = 1
    routing: dict[str, str] = {}  # the x-llmrouter-* headers of an auto-routed answer


class RunSummary(BaseModel):
    run: str
    items: int
    succeeded: int
    failed: int
    unpriced: int  # answers whose cost could not be worked out
    prompt_tokens: int
    completion_tokens: int
    total_cost_usd: float
    mean_cost_usd: float  # per successful, priced answer
    latency_p50_s: float | None
    latency_p95_s: float | None
    served_by: dict[str, int]


def percentile(values: list[float], fraction: float) -> float | None:
    """The nearest-rank percentile: the smallest value at least ``fraction`` of the data is <=."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def summarize(records: list[EvalRecord]) -> RunSummary:
    good = [r for r in records if r.ok]
    priced = [r for r in good if r.cost_usd is not None]
    served: dict[str, int] = {}
    for record in good:
        key = record.served_model or "unknown"
        served[key] = served.get(key, 0) + 1
    total = sum(r.cost_usd or 0.0 for r in priced)
    latencies = [r.latency_s for r in good if r.latency_s is not None]
    return RunSummary(
        run=records[0].run if records else "",
        items=len(records),
        succeeded=len(good),
        failed=len(records) - len(good),
        unpriced=len(good) - len(priced),
        prompt_tokens=sum(r.prompt_tokens or 0 for r in good),
        completion_tokens=sum(r.completion_tokens or 0 for r in good),
        total_cost_usd=total,
        mean_cost_usd=total / len(priced) if priced else 0.0,
        latency_p50_s=percentile(latencies, 0.50),
        latency_p95_s=percentile(latencies, 0.95),
        served_by=dict(sorted(served.items())),
    )


def load_records(path: Path) -> list[EvalRecord]:
    """Read a results file. If an item appears more than once the last record wins."""
    latest: dict[str, EvalRecord] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = EvalRecord.model_validate(json.loads(line))
        except ValueError as exc:
            raise ValueError(f"{path}, line {number}: {exc}") from None
        latest[record.item_id] = record
    return list(latest.values())


def _error_text(response: httpx.Response) -> str:
    try:
        message = response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        message = response.text
    return f"HTTP {response.status_code}: {str(message)[:300]}"


def _routing_headers(response: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in response.headers.items() if k.lower().startswith("x-llmrouter-")}


class Runner:
    """Sends benchmark items to an OpenAI-compatible endpoint and records what came back.

    The same code measures every strategy: ``model`` is the frontier alias for the always-frontier
    baseline and ``auto`` for the router, so the two differ in nothing but the model asked for.
    Transient failures (429 and 5xx) are retried, so a hiccup does not decide a result, but an
    item that still fails is recorded as a failure, never skipped, and a run with failures is
    reported as such.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        pricing: PriceBook,
        *,
        concurrency: int = 4,
        retries: int = 3,
        max_wait: float = 30.0,
        sleep: Sleep = asyncio.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if retries < 0:
            raise ValueError("retries must not be negative")
        self._client = client
        self._pricing = pricing
        self._concurrency = concurrency
        self._retries = retries
        self._max_wait = max_wait
        self._sleep = sleep
        self._clock = clock

    async def run(
        self,
        benchmark: Benchmark,
        *,
        model: str,
        label: str,
        out: Path | None = None,
        resume: bool = False,
        on_record: Callable[[EvalRecord], None] | None = None,
    ) -> list[EvalRecord]:
        """Ask every item of the benchmark, in benchmark order, and return the records.

        With ``out`` each record is appended to the file the moment it is known, so a crash
        loses nothing. With ``resume`` the items already answered successfully in that file are
        not asked again.
        """
        done: dict[str, EvalRecord] = {}
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            if resume and out.exists():
                done = {r.item_id: r for r in load_records(out) if r.ok and r.run == label}
            elif not resume:
                out.write_text("", encoding="utf-8")
        pending = [item for item in benchmark if item.id not in done]
        gate = asyncio.Semaphore(self._concurrency)
        write_lock = asyncio.Lock()
        results: dict[str, EvalRecord] = dict(done)

        async def one(item: BenchmarkItem) -> None:
            async with gate:
                record = await self.ask(item, model=model, label=label)
            results[item.id] = record
            if out is not None:
                async with write_lock:
                    with out.open("a", encoding="utf-8") as handle:
                        handle.write(record.model_dump_json() + "\n")
            if on_record is not None:
                on_record(record)

        await asyncio.gather(*(one(item) for item in pending))
        return [results[item.id] for item in benchmark]

    async def ask(self, item: BenchmarkItem, *, model: str, label: str) -> EvalRecord:
        """Send one item, retrying transient failures, and describe the outcome."""
        body = item.to_request(model).model_dump(exclude_none=True)
        attempts = 0
        while True:
            attempts += 1
            started = self._clock()
            try:
                response = await self._client.post("/v1/chat/completions", json=body)
            except httpx.HTTPError as exc:
                if attempts <= self._retries:
                    await self._sleep(min(2.0 ** (attempts - 1), self._max_wait))
                    continue
                return self._failure(
                    item, model, label, None, f"{type(exc).__name__}: {exc}", attempts
                )
            elapsed = self._clock() - started
            if response.status_code in RETRYABLE and attempts <= self._retries:
                await self._sleep(self._wait(response, attempts))
                continue
            if response.status_code != 200:
                return self._failure(
                    item, model, label, response.status_code, _error_text(response), attempts
                )
            return self._success(item, model, label, response, elapsed, attempts)

    def _wait(self, response: httpx.Response, attempts: int) -> float:
        try:
            asked = float(response.headers["Retry-After"])
        except (KeyError, ValueError):
            asked = 0.0
        return min(max(asked, 2.0 ** (attempts - 1)), self._max_wait)

    @staticmethod
    def _failure(
        item: BenchmarkItem,
        model: str,
        label: str,
        status: int | None,
        error: str,
        attempts: int,
    ) -> EvalRecord:
        return EvalRecord(
            item_id=item.id,
            run=label,
            requested_model=model,
            ok=False,
            status_code=status,
            error=error,
            attempts=attempts,
        )

    def _success(
        self,
        item: BenchmarkItem,
        model: str,
        label: str,
        response: httpx.Response,
        elapsed: float,
        attempts: int,
    ) -> EvalRecord:
        try:
            payload: dict[str, Any] = response.json()
            reply = payload["choices"][0]["message"].get("content")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            return self._failure(item, model, label, 200, "malformed completion response", attempts)
        routing = _routing_headers(response)
        served = routing.get("x-llmrouter-model") or (model if model in self._pricing else None)
        usage = payload.get("usage") or {}
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        cost: float | None = None
        if (
            served is not None
            and served in self._pricing
            and prompt is not None
            and completion is not None
        ):
            cost = self._pricing.cost(served, int(prompt), int(completion))
        return EvalRecord(
            item_id=item.id,
            run=label,
            requested_model=model,
            served_model=served,
            ok=reply is not None,
            status_code=200,
            error=None if reply is not None else "the model returned no content",
            reply=reply,
            prompt_tokens=prompt,
            completion_tokens=completion,
            cost_usd=cost,
            latency_s=round(elapsed, 6),
            attempts=attempts,
            routing=routing,
        )
