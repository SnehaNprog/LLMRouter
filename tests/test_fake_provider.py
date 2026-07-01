import random

import pytest

from app.providers.errors import AuthFailed, ProviderDown, ProviderError
from app.providers.fake import FakeProvider
from app.schemas.chat import ChatCompletionRequest, Message

REQUEST = ChatCompletionRequest(model="m", messages=[Message(role="user", content="hi")])


async def test_chat_returns_configured_reply_and_tokens() -> None:
    provider = FakeProvider(reply="pong", prompt_tokens=7, completion_tokens=3)

    response = await provider.chat(REQUEST)

    assert response.choices[0].message.content == "pong"
    assert response.usage is not None
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (7, 3)
    assert response.usage.total_tokens == 10
    assert response.model == "m"


async def test_records_calls_and_requests() -> None:
    provider = FakeProvider()

    await provider.chat(REQUEST)
    await provider.chat(REQUEST)

    assert provider.calls == 2
    assert provider.requests == [REQUEST, REQUEST]


async def test_latency_uses_injected_sleep() -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    await FakeProvider(latency=0.25, sleep=fake_sleep).chat(REQUEST)

    assert slept == [0.25]


async def test_fail_first_then_succeeds() -> None:
    provider = FakeProvider(fail_first=2)

    for _ in range(2):
        with pytest.raises(ProviderDown):
            await provider.chat(REQUEST)
    response = await provider.chat(REQUEST)

    assert response.choices[0].message.content
    assert provider.calls == 3


async def test_custom_error_factory() -> None:
    provider = FakeProvider(fail_first=1, error_factory=lambda: AuthFailed("bad key"))

    with pytest.raises(AuthFailed):
        await provider.chat(REQUEST)


async def test_failure_rate_is_repeatable_with_seeded_rng() -> None:
    async def outcomes() -> list[bool]:
        provider = FakeProvider(failure_rate=0.5, rng=random.Random(42))
        results = []
        for _ in range(20):
            try:
                await provider.chat(REQUEST)
                results.append(True)
            except ProviderError:
                results.append(False)
        return results

    first, second = await outcomes(), await outcomes()

    assert first == second
    assert True in first
    assert False in first


async def test_failure_rate_extremes() -> None:
    always = FakeProvider(failure_rate=1.0)
    never = FakeProvider(failure_rate=0.0)

    with pytest.raises(ProviderError):
        await always.chat(REQUEST)
    await never.chat(REQUEST)


async def test_stream_yields_role_words_then_finish() -> None:
    provider = FakeProvider(reply="one two three")

    chunks = [chunk async for chunk in provider.stream(REQUEST)]

    assert chunks[0].choices[0].delta.role == "assistant"
    text = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert text == "one two three"
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert all(c.choices[0].finish_reason is None for c in chunks[:-1])


async def test_stream_failure_raises_before_any_chunk() -> None:
    provider = FakeProvider(fail_first=1)
    received = []

    with pytest.raises(ProviderDown):
        async for chunk in provider.stream(REQUEST):
            received.append(chunk)

    assert received == []
