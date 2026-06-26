from collections.abc import AsyncIterator

import pytest

from app.providers.base import Provider
from app.schemas.chat import ChatCompletionRequest, Message
from app.schemas.completion import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    Choice,
    ChunkChoice,
    Delta,
    ResponseMessage,
)

REQUEST = ChatCompletionRequest(model="m", messages=[Message(role="user", content="hi")])


class EchoProvider(Provider):
    name = "echo"

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id="1",
            created=0,
            model=request.model,
            choices=[Choice(index=0, message=ResponseMessage(content="hi"), finish_reason="stop")],
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        yield ChatCompletionChunk(
            id="1",
            created=0,
            model=request.model,
            choices=[ChunkChoice(index=0, delta=Delta(content="hi"))],
        )
        yield ChatCompletionChunk(
            id="1",
            created=0,
            model=request.model,
            choices=[ChunkChoice(index=0, delta=Delta(), finish_reason="stop")],
        )


def test_provider_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Provider()  # type: ignore[abstract]


def test_incomplete_subclass_is_rejected() -> None:
    class OnlyChat(Provider):
        name = "x"

        async def chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
            raise NotImplementedError

    with pytest.raises(TypeError):
        OnlyChat()  # type: ignore[abstract]


async def test_concrete_provider_satisfies_the_interface() -> None:
    provider = EchoProvider()

    response = await provider.chat(REQUEST)
    chunks = [chunk async for chunk in provider.stream(REQUEST)]

    assert response.choices[0].message.content == "hi"
    assert chunks[-1].choices[0].finish_reason == "stop"
