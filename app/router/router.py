import asyncio
import logging
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

from app.router.deployment import Deployment
from app.router.registry import ModelRegistry
from app.router.retry import Decision, RetryPolicy, backoff_delay, classify
from app.router.selection import weighted_choice
from app.schemas.chat import ChatCompletionRequest
from app.schemas.completion import ChatCompletionChunk, ChatCompletionResponse

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]


class Router:
    """Sends a request to a deployment of the requested alias, retrying on failure.

    Deployments are picked at random in proportion to their weight. After a failure the
    router prefers a deployment it has not tried yet and moves on immediately; only when it
    must hit a deployment again does it wait, with exponential backoff and jitter.
    When every retry on an alias has failed, the next alias in its fallback chain is tried.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        policy: RetryPolicy | None = None,
        rng: random.Random | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.registry = registry
        self.policy = policy or RetryPolicy()
        self._rng = rng or random.Random()
        self._sleep = sleep

    def select(self, alias: str) -> Deployment:
        return weighted_choice(self.registry.get(alias), self._rng)

    @staticmethod
    def _upstream_request(
        request: ChatCompletionRequest, deployment: Deployment
    ) -> ChatCompletionRequest:
        return request.model_copy(update={"model": deployment.upstream_model})

    def _pick(self, deployments: Sequence[Deployment], tried: set[str]) -> Deployment:
        untried = [d for d in deployments if d.id not in tried]
        return weighted_choice(untried or deployments, self._rng)

    async def _with_retries[T](
        self,
        deployments: Sequence[Deployment],
        attempt: Callable[[Deployment], Awaitable[T]],
    ) -> T:
        tried: set[str] = set()
        attempt_number = 0
        while True:
            attempt_number += 1
            deployment = self._pick(deployments, tried)
            tried.add(deployment.id)
            try:
                return await attempt(deployment)
            except Exception as exc:
                decision = classify(exc)
                if decision is Decision.FAIL or attempt_number >= self.policy.max_attempts:
                    raise
                has_untried = any(d.id not in tried for d in deployments)
                delay = (
                    0.0
                    if has_untried
                    else backoff_delay(
                        attempt_number,
                        self.policy,
                        self._rng,
                        retry_after=getattr(exc, "retry_after", None),
                    )
                )
                logger.warning(
                    "attempt failed, retrying",
                    extra={
                        "deployment": deployment.id,
                        "attempt": attempt_number,
                        "error": type(exc).__name__,
                        "decision": decision.value,
                        "delay": round(delay, 3),
                    },
                )
                if delay > 0:
                    await self._sleep(delay)

    async def _across_fallbacks[T](
        self, alias: str, run: Callable[[Sequence[Deployment]], Awaitable[T]]
    ) -> T:
        chain = self.registry.chain(alias)
        for position, name in enumerate(chain):
            try:
                return await run(self.registry.get(name))
            except Exception as exc:
                if position == len(chain) - 1 or classify(exc) is Decision.FAIL:
                    raise
                logger.warning(
                    "model exhausted, falling back",
                    extra={
                        "model": name,
                        "fallback": chain[position + 1],
                        "error": type(exc).__name__,
                    },
                )
        raise AssertionError("unreachable: the chain always has at least one alias")

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        async def attempt(deployment: Deployment) -> ChatCompletionResponse:
            return await deployment.provider.chat(self._upstream_request(request, deployment))

        async def run(deployments: Sequence[Deployment]) -> ChatCompletionResponse:
            return await self._with_retries(deployments, attempt)

        return await self._across_fallbacks(request.model, run)

    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        # Unknown aliases raise here, before any iteration, so the API can answer with a 404.
        self.registry.get(request.model)
        return self._stream(request)

    async def _stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        async def open_stream(
            deployment: Deployment,
        ) -> tuple[ChatCompletionChunk | None, AsyncIterator[ChatCompletionChunk]]:
            stream = deployment.provider.stream(self._upstream_request(request, deployment))
            try:
                return await anext(stream), stream
            except StopAsyncIteration:
                return None, stream

        async def run(
            deployments: Sequence[Deployment],
        ) -> tuple[ChatCompletionChunk | None, AsyncIterator[ChatCompletionChunk]]:
            return await self._with_retries(deployments, open_stream)

        # Only failures before the first chunk are retried or fall back; once output has
        # started the client has seen it, so later errors propagate.
        first, rest = await self._across_fallbacks(request.model, run)
        if first is not None:
            yield first
        async for chunk in rest:
            yield chunk
