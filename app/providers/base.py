from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.schemas.chat import ChatCompletionRequest
from app.schemas.completion import ChatCompletionChunk, ChatCompletionResponse


class Provider(ABC):
    """An upstream LLM vendor behind the gateway.

    Adapters translate our OpenAI-shaped request into the vendor's wire format
    and translate the reply back. The router sets ``request.model`` to the
    vendor's own model name before calling an adapter.

    Adapters must raise the domain errors from ``app.providers.errors`` and
    never leak vendor-specific exceptions to the router.
    """

    name: str

    @abstractmethod
    async def chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Run one non-streaming completion."""

    @abstractmethod
    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        """Run a streaming completion.

        Implement as an async generator (``async def`` with ``yield``). Yield
        chunks in order; the last one carries ``finish_reason``. Errors that occur
        before the first chunk must raise, so the router can still retry or
        fall back.
        """
