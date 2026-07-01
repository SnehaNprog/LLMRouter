import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable

from app.providers.base import Provider
from app.providers.errors import ProviderDown, ProviderError
from app.schemas.chat import ChatCompletionRequest
from app.schemas.completion import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    Choice,
    ChunkChoice,
    Delta,
    ResponseMessage,
    Usage,
)


class FakeProvider(Provider):
    """Deterministic stand-in for a real vendor, used by router and quota tests.

    ``fail_first`` makes the first N calls fail, then succeed (retry tests).
    ``failure_rate`` fails a random share of calls; pass a seeded ``rng`` to make
    that repeatable. ``sleep`` is injectable so tests need no real waiting.
    """

    def __init__(
        self,
        name: str = "fake",
        *,
        reply: str = "Hello from the fake provider",
        latency: float = 0.0,
        failure_rate: float = 0.0,
        fail_first: int = 0,
        error_factory: Callable[[], ProviderError] | None = None,
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        rng: random.Random | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.name = name
        self.reply = reply
        self.latency = latency
        self.failure_rate = failure_rate
        self.fail_first = fail_first
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self._error_factory = error_factory or (
            lambda: ProviderDown("fake provider failure", provider=name)
        )
        self._rng = rng or random.Random(0)
        self._sleep = sleep
        self.calls = 0
        self.requests: list[ChatCompletionRequest] = []

    async def _begin(self, request: ChatCompletionRequest) -> None:
        self.calls += 1
        self.requests.append(request)
        if self.latency:
            await self._sleep(self.latency)
        if self.calls <= self.fail_first or self._rng.random() < self.failure_rate:
            raise self._error_factory()

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        await self._begin(request)
        return ChatCompletionResponse(
            id=f"chatcmpl-{self.name}-{self.calls}",
            created=0,
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ResponseMessage(content=self.reply),
                    finish_reason="stop",
                )
            ],
            usage=Usage.of(self.prompt_tokens, self.completion_tokens),
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        await self._begin(request)
        chunk_id = f"chatcmpl-{self.name}-{self.calls}"

        def chunk(delta: Delta, finish: bool = False) -> ChatCompletionChunk:
            return ChatCompletionChunk(
                id=chunk_id,
                created=0,
                model=request.model,
                choices=[
                    ChunkChoice(index=0, delta=delta, finish_reason="stop" if finish else None)
                ],
            )

        yield chunk(Delta(role="assistant", content=""))
        words = self.reply.split(" ")
        for position, word in enumerate(words):
            yield chunk(Delta(content=word if position == 0 else f" {word}"))
        yield chunk(Delta(), finish=True)
