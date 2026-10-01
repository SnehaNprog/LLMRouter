import asyncio
import json
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field


@dataclass
class StubSettings:
    """How the stand-in model behaves."""

    latency_ms: float = 50.0  # time to a full answer (or to the first chunk when streaming)
    jitter_ms: float = 0.0  # each answer takes latency_ms plus or minus up to this much
    error_rate: float = 0.0  # share of requests answered with a 503
    throttle_rate: float = 0.0  # share answered with a 429 and a Retry-After
    prompt_tokens: int = 20
    completion_tokens: int = 50
    stream_chunks: int = 8  # content chunks in a streamed answer
    chunk_gap_ms: float = 5.0
    seed: int | None = 1


class StubControl(BaseModel):
    """Settings that can be changed while the stub runs (to start or stop an outage)."""

    latency_ms: float | None = Field(default=None, ge=0)
    error_rate: float | None = Field(default=None, ge=0, le=1)
    throttle_rate: float | None = Field(default=None, ge=0, le=1)


class Counters:
    def __init__(self) -> None:
        self.requests = 0
        self.errors = 0
        self.throttled = 0
        self.in_flight = 0
        self.peak_in_flight = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "requests": self.requests,
            "errors": self.errors,
            "throttled": self.throttled,
            "in_flight": self.in_flight,
            "peak_in_flight": self.peak_in_flight,
        }


def create_stub(settings: StubSettings | None = None) -> FastAPI:
    """An OpenAI-compatible server that answers after a chosen delay and fails on demand.

    It stands in for a real provider, so the gateway can be loaded hard without spending money or
    being throttled, and so what the upstream did can be told apart from what the gateway did:
    ``/__stub/stats`` says how many requests actually arrived.
    """
    config = settings or StubSettings()
    rng = random.Random(config.seed)
    counters = Counters()
    app = FastAPI(title="LLMRouter load-test stub", docs_url=None, redoc_url=None)

    def delay_seconds() -> float:
        jitter = rng.uniform(-config.jitter_ms, config.jitter_ms) if config.jitter_ms else 0.0
        return max(0.0, config.latency_ms + jitter) / 1000

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/__stub/stats")
    def stats() -> dict[str, int]:
        return counters.snapshot()

    @app.post("/__stub/config")
    def reconfigure(change: StubControl) -> dict[str, float]:
        for name, value in change.model_dump(exclude_none=True).items():
            setattr(config, name, value)
        return {
            "latency_ms": config.latency_ms,
            "error_rate": config.error_rate,
            "throttle_rate": config.throttle_rate,
        }

    @app.post("/__stub/reset")
    def reset() -> dict[str, int]:
        peak = counters.snapshot()
        counters.requests = counters.errors = counters.throttled = counters.peak_in_flight = 0
        return peak

    @app.post("/v1/chat/completions", response_model=None)
    async def completions(request: Request) -> Response:
        body = await request.json()
        counters.requests += 1
        counters.in_flight += 1
        counters.peak_in_flight = max(counters.peak_in_flight, counters.in_flight)
        try:
            await asyncio.sleep(delay_seconds())
            roll = rng.random()
            if roll < config.error_rate:
                counters.errors += 1
                return JSONResponse(
                    {"error": {"message": "stub failure", "type": "server_error"}}, status_code=503
                )
            if roll < config.error_rate + config.throttle_rate:
                counters.throttled += 1
                return JSONResponse(
                    {"error": {"message": "stub throttle", "type": "rate_limit_error"}},
                    status_code=429,
                    headers={"Retry-After": "1"},
                )
            model = str(body.get("model", "stub"))
            if body.get("stream"):
                return StreamingResponse(
                    _stream(model, config, counters), media_type="text/event-stream"
                )
            return JSONResponse(_completion(model, config))
        finally:
            if not body.get("stream"):
                counters.in_flight -= 1

    return app


def _usage(config: StubSettings) -> dict[str, int]:
    return {
        "prompt_tokens": config.prompt_tokens,
        "completion_tokens": config.completion_tokens,
        "total_tokens": config.prompt_tokens + config.completion_tokens,
    }


def _completion(model: str, config: StubSettings) -> dict[str, object]:
    return {
        "id": "stub-completion",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "A stubbed answer."},
                "finish_reason": "stop",
            }
        ],
        "usage": _usage(config),
    }


async def _stream(model: str, config: StubSettings, counters: Counters) -> AsyncIterator[str]:
    def chunk(delta: dict[str, str], finish: str | None = None, usage: bool = False) -> str:
        payload: dict[str, object] = {
            "id": "stub-chunk",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": ([] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}]),
        }
        if usage:
            payload["usage"] = _usage(config)
        return f"data: {json.dumps(payload)}\n\n"

    try:
        yield chunk({"role": "assistant", "content": ""})
        for n in range(config.stream_chunks):
            await asyncio.sleep(config.chunk_gap_ms / 1000)
            yield chunk({"content": f"word{n} "})
        yield chunk({}, finish="stop")
        yield chunk({}, usage=True)
        yield "data: [DONE]\n\n"
    finally:
        counters.in_flight -= 1
