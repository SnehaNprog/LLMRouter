from collections.abc import Callable
from urllib.parse import urlparse

import httpx

from app.core.model_config import ConfigError, ModelConfig
from app.providers.anthropic import AnthropicProvider
from app.providers.base import Provider
from app.providers.openai_compat import OpenAICompatibleProvider

Builder = Callable[[ModelConfig, httpx.AsyncClient], Provider]


def _build_openai_compatible(config: ModelConfig, client: httpx.AsyncClient) -> Provider:
    assert config.base_url is not None  # enforced when the config is loaded
    # Named after the host (api.groq.com, localhost) so every deployment on the same vendor
    # shares one identity, which per-provider circuit breakers rely on.
    name = urlparse(config.base_url).hostname or "openai_compatible"
    api_key = config.api_key.get_secret_value() if config.api_key else None
    return OpenAICompatibleProvider(name, base_url=config.base_url, api_key=api_key, client=client)


def _build_anthropic(config: ModelConfig, client: httpx.AsyncClient) -> Provider:
    if config.api_key is None:
        raise ConfigError(f"model '{config.name}': provider 'anthropic' requires an api_key")
    api_key = config.api_key.get_secret_value()
    if config.base_url is None:
        return AnthropicProvider("anthropic", api_key=api_key, client=client)
    return AnthropicProvider("anthropic", api_key=api_key, client=client, base_url=config.base_url)


_BUILDERS: dict[str, Builder] = {
    "openai_compatible": _build_openai_compatible,
    "anthropic": _build_anthropic,
}


def known_providers() -> list[str]:
    return sorted(_BUILDERS)


def build_provider(config: ModelConfig, client: httpx.AsyncClient) -> Provider:
    builder = _BUILDERS.get(config.provider)
    if builder is None:
        raise ConfigError(
            f"model '{config.name}': unknown provider '{config.provider}'; "
            f"known providers: {', '.join(known_providers())}"
        )
    return builder(config, client)
