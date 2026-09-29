import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from functools import partial

import httpx
from fastapi import FastAPI

from app.core.config import Settings, provider_environment
from app.core.http import build_http_client
from app.core.model_config import ConfigError, load_config
from app.core.redis import build_redis_client, redis_ready
from app.db.session import create_engine, create_session_factory, database_ready
from app.quota.bucket import BucketStore
from app.quota.budget import BudgetStore
from app.quota.budget_limiter import BudgetLimiter
from app.quota.guard import QuotaGuard
from app.quota.limiter import RateLimiter
from app.resilience.board import BreakerBoard, BreakerSettings
from app.router.latency import LatencyTracker
from app.router.registry import ModelRegistry, build_registry
from app.router.retry import RetryPolicy
from app.router.router import Router
from app.routing.score import ComplexityWeights

logger = logging.getLogger(__name__)

RegistryFactory = Callable[[httpx.AsyncClient], ModelRegistry]


def default_registry_factory(settings: Settings) -> RegistryFactory:
    def factory(client: httpx.AsyncClient) -> ModelRegistry:
        config = load_config(settings.config_path, provider_environment())
        return build_registry(
            config.models,
            client,
            config.fallbacks,
            ComplexityWeights.from_overrides(config.complexity),
            config.policy,
        )

    return factory


def make_lifespan(
    settings: Settings, registry_factory: RegistryFactory
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Load the model config (failing fast on a bad one), share one HTTP client, clean up."""
        client = build_http_client(settings)
        app.state.http_client = client
        engine = create_engine(settings.database_url)
        app.state.db_engine = engine
        app.state.db_session_factory = create_session_factory(engine)
        app.state.readiness_checks["database"] = partial(database_ready, engine)
        redis = build_redis_client(settings)  # lazy: Redis being down never blocks startup
        app.state.redis = redis
        guard = QuotaGuard(
            settings.quota_failure_mode,
            failure_threshold=settings.quota_breaker_threshold,
            cooldown=settings.quota_breaker_cooldown,
        )
        app.state.quota_guard = guard
        # Failing open, the gateway still works without Redis, so Redis being down must not
        # take its pods out of rotation; failing closed, limited keys cannot be served without it.
        app.state.optional_checks = {"redis"} if guard.mode == "open" else set()
        app.state.budgets = BudgetLimiter(
            BudgetStore(redis, settings.redis_key_prefix + "budget:"),
            completion_reserve=settings.default_completion_reserve,
            guard=guard,
        )
        app.state.limiter = RateLimiter(
            BucketStore(redis, settings.redis_key_prefix + "quota:"),
            default_rpm=settings.default_rpm_limit,
            default_tpm=settings.default_tpm_limit,
            completion_reserve=settings.default_completion_reserve,
            guard=guard,
        )
        app.state.readiness_checks["redis"] = partial(redis_ready, redis)
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
                breakers=BreakerBoard(
                    BreakerSettings(
                        failure_threshold=settings.breaker_failure_threshold or None,
                        failure_rate_threshold=settings.breaker_failure_rate or None,
                        min_requests=settings.breaker_min_requests,
                        window_seconds=settings.breaker_window_seconds,
                        cooldown=settings.breaker_cooldown,
                    )
                ),
            )
            logger.info(
                "model config loaded",
                extra={"aliases": registry.aliases(), "deployments": len(registry)},
            )
            yield
        finally:
            await client.aclose()
            await redis.aclose()
            await engine.dispose()

    return lifespan
