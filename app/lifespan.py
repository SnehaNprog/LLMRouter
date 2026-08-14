import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import httpx
from fastapi import FastAPI

from app.core.config import Settings, provider_environment
from app.core.http import build_http_client
from app.core.model_config import ConfigError, load_config
from app.router.latency import LatencyTracker
from app.router.registry import ModelRegistry, build_registry
from app.router.retry import RetryPolicy
from app.router.router import Router

logger = logging.getLogger(__name__)

RegistryFactory = Callable[[httpx.AsyncClient], ModelRegistry]


def default_registry_factory(settings: Settings) -> RegistryFactory:
    def factory(client: httpx.AsyncClient) -> ModelRegistry:
        config = load_config(settings.config_path, provider_environment())
        return build_registry(config.models, client, config.fallbacks)

    return factory


def make_lifespan(
    settings: Settings, registry_factory: RegistryFactory
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Load the model config (failing fast on a bad one), share one HTTP client, clean up."""
        client = build_http_client(settings)
        app.state.http_client = client
        try:
            try:
                registry = registry_factory(client)
            except ConfigError as exc:
                logger.error("startup aborted: %s", exc)
                raise
            app.state.registry = registry
            app.state.router = Router(
                registry,
                policy=RetryPolicy(
                    max_attempts=settings.retry_max_attempts,
                    base_delay=settings.retry_base_delay,
                    max_delay=settings.retry_max_delay,
                ),
                latency=LatencyTracker(settings.latency_ewma_alpha),
            )
            logger.info(
                "model config loaded",
                extra={"aliases": registry.aliases(), "deployments": len(registry)},
            )
            yield
        finally:
            await client.aclose()

    return lifespan
