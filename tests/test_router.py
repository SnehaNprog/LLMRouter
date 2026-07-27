import pytest

from app.providers.fake import FakeProvider
from app.router.registry import UnknownModelError
from app.router.router import Router
from app.schemas.chat import ChatCompletionRequest, Message
from tests.conftest import TEST_ALIAS, UPSTREAM_MODEL, registry_with


def request(model: str = TEST_ALIAS) -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model, messages=[Message(role="user", content="hi")])


async def test_complete_calls_the_provider_with_the_upstream_model() -> None:
    provider = FakeProvider(reply="pong")
    router = Router(registry_with(provider))

    response = await router.complete(request())

    assert response.choices[0].message.content == "pong"
    assert provider.requests[0].model == UPSTREAM_MODEL
    assert provider.requests[0].messages == request().messages


async def test_original_request_is_not_mutated() -> None:
    original = request()

    await Router(registry_with(FakeProvider())).complete(original)

    assert original.model == TEST_ALIAS


async def test_unknown_alias_raises() -> None:
    router = Router(registry_with(FakeProvider()))

    with pytest.raises(UnknownModelError):
        await router.complete(request("nope"))


async def test_stream_uses_the_upstream_model() -> None:
    provider = FakeProvider(reply="a b")
    router = Router(registry_with(provider))

    chunks = [chunk async for chunk in router.stream(request())]

    assert provider.requests[0].model == UPSTREAM_MODEL
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_stream_with_unknown_alias_raises_before_iterating() -> None:
    router = Router(registry_with(FakeProvider()))

    with pytest.raises(UnknownModelError):
        router.stream(request("nope"))


def test_select_returns_the_first_deployment() -> None:
    router = Router(registry_with(FakeProvider()))

    assert router.select(TEST_ALIAS).alias == TEST_ALIAS
