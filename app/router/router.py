import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass

from app.router.context import ContextLengthExceededError, fits
from app.router.deployment import Deployment
from app.router.latency import LatencyTracker
from app.router.registry import ModelRegistry
from app.router.retry import Decision, RetryPolicy, backoff_delay, classify
from app.router.selection import weighted_choice
from app.router.tokens import estimate_prompt_tokens
from app.schemas.chat import ChatCompletionRequest
from app.schemas.completion import ChatCompletionChunk, ChatCompletionResponse, Usage

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]
Admit = Callable[[str], Awaitable[bool]]


@dataclass
class Served:
    """Filled in by the router with the deployment that actually answered a request.

    Retries and fallbacks decide this late, and the caller needs it to price the request.
    """

    deployment: Deployment | None = None


class Router:
    """Sends a request to a deployment of the requested alias, retrying on failure.

    Deployments are picked at random in proportion to their weight. After a failure the
    router prefers a deployment it has not tried yet and moves on immediately; only when it
    must hit a deployment again does it wait, with exponential backoff and jitter.
    When every retry on an alias has failed, the next alias in its fallback chain is tried.
    Successful calls feed a per-deployment latency tracker (time to first chunk when streaming).
    """

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        policy: RetryPolicy | None = None,
        rng: random.Random | None = None,
        sleep: Sleep = asyncio.sleep,
        latency: LatencyTracker | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.registry = registry
        self.policy = policy or RetryPolicy()
        self.latency = latency or LatencyTracker()
        self._rng = rng or random.Random()
        self._sleep = sleep
        self._clock = clock

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
        self,
        request: ChatCompletionRequest,
        run: Callable[[Sequence[Deployment]], Awaitable[T]],
        escalation: Sequence[str] = (),
        admit: Admit | None = None,
    ) -> T:
        """Run ``run`` on the request's alias, then on its fallbacks, until one succeeds.

        After those, the ``escalation`` models are tried in order (each with its own fallbacks).
        ``admit`` is asked before each of them whether the request may go there, and a model it
        refuses is skipped.

        Deployments whose context window cannot hold the request are skipped; an alias with
        none left is passed over without being called.
        """
        required = estimate_prompt_tokens(request) + (request.max_tokens or 0)
        chain = self.registry.chain(request.model)
        configured = len(chain)
        for alias in escalation:
            chain.extend(name for name in self.registry.chain(alias) if name not in chain)
        last_error: Exception | None = None
        largest_window = 0
        for position, name in enumerate(chain):
            if position >= configured and admit is not None and not await admit(name):
                logger.info("escalation target skipped, not admitted", extra={"model": name})
                continue
            available = self.registry.get(name)
            candidates = [d for d in available if fits(d, required)]
            if not candidates:
                largest_window = max(largest_window, *(d.context_window or 0 for d in available))
                logger.info(
                    "model skipped, request exceeds its context window",
                    extra={"model": name, "required_tokens": required},
                )
                continue
            try:
                return await run(candidates)
            except Exception as exc:
                if classify(exc) is Decision.FAIL:
                    raise
                last_error = exc
                if position < len(chain) - 1:
                    logger.warning(
                        "model exhausted, falling back",
                        extra={
                            "model": name,
                            "fallback": chain[position + 1],
                            "error": type(exc).__name__,
                        },
                    )
        if last_error is not None:
            raise last_error
        raise ContextLengthExceededError(window=largest_window, required=required)

    @staticmethod
    def _log_completion(deployment: Deployment, usage: Usage | None) -> None:
        if usage is None:
            return
        logger.info(
            "completion",
            extra={
                "deployment": deployment.id,
                "model": deployment.upstream_model,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cost_usd": deployment.cost(usage.prompt_tokens, usage.completion_tokens),
            },
        )

    async def complete(
        self,
        request: ChatCompletionRequest,
        served: Served | None = None,
        *,
        escalation: Sequence[str] = (),
        admit: Admit | None = None,
    ) -> ChatCompletionResponse:
        async def attempt(
            deployment: Deployment,
        ) -> tuple[Deployment, ChatCompletionResponse]:
            started = self._clock()
            response = await deployment.provider.chat(self._upstream_request(request, deployment))
            self.latency.observe(deployment.id, self._clock() - started)
            return deployment, response

        async def run(
            deployments: Sequence[Deployment],
        ) -> tuple[Deployment, ChatCompletionResponse]:
            return await self._with_retries(deployments, attempt)

        deployment, response = await self._across_fallbacks(request, run, escalation, admit)
        if served is not None:
            served.deployment = deployment
        self._log_completion(deployment, response.usage)
        return response

    def stream(
        self,
        request: ChatCompletionRequest,
        served: Served | None = None,
        *,
        escalation: Sequence[str] = (),
        admit: Admit | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        # Unknown aliases raise here, before any iteration, so the API can answer with a 404.
        self.registry.get(request.model)
        return self._stream(request, served, escalation, admit)

    async def _stream(
        self,
        request: ChatCompletionRequest,
        served: Served | None,
        escalation: Sequence[str],
        admit: Admit | None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        opened = tuple[Deployment, ChatCompletionChunk | None, AsyncIterator[ChatCompletionChunk]]

        async def open_stream(deployment: Deployment) -> opened:
            started = self._clock()
            stream = deployment.provider.stream(self._upstream_request(request, deployment))
            try:
                first = await anext(stream)
            except StopAsyncIteration:
                return deployment, None, stream
            # For streams the latency that matters to the caller is time to first chunk.
            self.latency.observe(deployment.id, self._clock() - started)
            return deployment, first, stream

        async def run(deployments: Sequence[Deployment]) -> opened:
            return await self._with_retries(deployments, open_stream)

        # Only failures before the first chunk are retried or fall back; once output has
        # started the client has seen it, so later errors propagate.
        deployment, first, rest = await self._across_fallbacks(request, run, escalation, admit)
        if served is not None:
            served.deployment = deployment
        if first is not None:
            self._log_completion(deployment, first.usage)
            yield first
        async for chunk in rest:
            self._log_completion(deployment, chunk.usage)
            yield chunk
