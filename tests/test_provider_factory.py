from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from app.core.model_config import ConfigError, ModelConfig
from app.providers.anthropic import AnthropicProvider
from app.providers.factory import build_provider, known_providers
from app.providers.openai_compat import OpenAICompatibleProvider
from app.schemas.chat import ChatCompletionRequest, Message

REQUEST = ChatCompletionRequest(model="m", messages=[Message(role="user", content="hi")])


def config(**overrides: Any) -> ModelConfig:
    fields: dict[str, Any] = {
        "name": "fast",
        "provider": "openai_compatible",
        "model": "llama-3.1-8b-instant",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": SecretStr("gsk-secret"),
        "weight": 1,
        "rpm": None,
        "tpm": None,
    }
    fields.update(overrides)
    return ModelConfig(**fields)


def recording_client(seen: list[httpx.Request]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(500, json={"error": {"message": "x"}})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_builds_openai_compatible_provider() -> None:
    provider = build_provider(config(), httpx.AsyncClient())

    assert isinstance(provider, OpenAICompatibleProvider)


def test_openai_compatible_is_named_after_its_host() -> None:
    groq = build_provider(config(), httpx.AsyncClient())
    ollama = build_provider(
        config(base_url="http://localhost:11434/v1", api_key=None), httpx.AsyncClient()
    )

    assert groq.name == "api.groq.com"
    assert ollama.name == "localhost"


def test_builds_anthropic_provider() -> None:
    provider = build_provider(
        config(provider="anthropic", base_url=None, api_key=SecretStr("sk-ant")),
        httpx.AsyncClient(),
    )

    assert isinstance(provider, AnthropicProvider)
    assert provider.name == "anthropic"


def test_anthropic_without_key_is_a_clear_error() -> None:
    with pytest.raises(ConfigError, match="requires an api_key"):
        build_provider(
            config(provider="anthropic", base_url=None, api_key=None), httpx.AsyncClient()
        )


def test_unknown_provider_lists_the_known_ones() -> None:
    with pytest.raises(ConfigError) as excinfo:
        build_provider(config(provider="gemini"), httpx.AsyncClient())

    message = str(excinfo.value)
    assert "unknown provider 'gemini'" in message
    assert "'fast'" in message
    assert "anthropic" in message
    assert "openai_compatible" in message


def test_known_providers() -> None:
    assert known_providers() == ["anthropic", "openai_compatible"]


async def test_secret_and_base_url_reach_the_adapter() -> None:
    seen: list[httpx.Request] = []
    provider = build_provider(config(), recording_client(seen))

    with pytest.raises(Exception):  # noqa: B017 - upstream is stubbed to fail
        await provider.chat(REQUEST)

    assert str(seen[0].url) == "https://api.groq.com/openai/v1/chat/completions"
    assert seen[0].headers["Authorization"] == "Bearer gsk-secret"


async def test_anthropic_custom_base_url_is_used() -> None:
    seen: list[httpx.Request] = []
    provider = build_provider(
        config(provider="anthropic", base_url="https://proxy.example.com", api_key=SecretStr("k")),
        recording_client(seen),
    )

    with pytest.raises(Exception):  # noqa: B017 - upstream is stubbed to fail
        await provider.chat(REQUEST)

    assert str(seen[0].url) == "https://proxy.example.com/v1/messages"
    assert seen[0].headers["x-api-key"] == "k"
