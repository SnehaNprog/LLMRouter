import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

RETRYABLE = {429, 502, 503, 504}
Sleep = Callable[[float], Awaitable[None]]


@dataclass
class Attempt:
    """The outcome of posting a request, after any retries."""

    response: httpx.Response | None  # None if no response ever came back
    error: str | None  # what went wrong when there was no response
    attempts: int
    elapsed: float  # seconds the final attempt took


def error_text(response: httpx.Response) -> str:
    try:
        message = response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        message = response.text
    return f"HTTP {response.status_code}: {str(message)[:300]}"


def wait_for(response: httpx.Response, attempts: int, max_wait: float) -> float:
    """How long to wait before retrying: the server's Retry-After, or an exponential backoff."""
    try:
        asked = float(response.headers["Retry-After"])
    except (KeyError, ValueError):
        asked = 0.0
    return min(max(asked, 2.0 ** (attempts - 1)), max_wait)


async def post_with_retries(
    client: httpx.AsyncClient,
    body: dict[str, Any],
    *,
    retries: int,
    max_wait: float,
    sleep: Sleep = asyncio.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> Attempt:
    """POST a chat completion, retrying transient failures (429, 502-504, network errors).

    Returns the last response whatever its status, so the caller decides what a non-200 means.
    """
    attempts = 0
    while True:
        attempts += 1
        started = clock()
        try:
            response = await client.post("/v1/chat/completions", json=body)
        except httpx.HTTPError as exc:
            if attempts <= retries:
                await sleep(min(2.0 ** (attempts - 1), max_wait))
                continue
            return Attempt(None, f"{type(exc).__name__}: {exc}", attempts, 0.0)
        elapsed = clock() - started
        if response.status_code in RETRYABLE and attempts <= retries:
            await sleep(wait_for(response, attempts, max_wait))
            continue
        return Attempt(response, None, attempts, elapsed)
