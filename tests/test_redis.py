import asyncio
import os
from pathlib import Path
from typing import Any

import fakeredis
import httpx
import pytest
import yaml
from fastapi import FastAPI
from redis.asyncio import BlockingConnectionPool, Redis

from app.core.config import Settings
from app.core.redis import build_redis_client, redis_ready
from app.main import create_app
from app.providers.fake import FakeProvider
from app.router.registry import ModelRegistry
from tests.conftest import AuthedClient, registry_with

ROOT = Path(__file__).parent.parent

# ---------- what Redis must provide (run against the fake and, opt-in, a real server) ----------


async def test_strings_round_trip_as_text(redis_client: Redis) -> None:
    await redis_client.set("greeting", "hello")

    assert await redis_client.get("greeting") == "hello"


async def test_counters_are_atomic_under_concurrency(redis_client: Redis) -> None:
    await asyncio.gather(*(redis_client.incr("hits") for _ in range(200)))

    assert await redis_client.get("hits") == "200"


async def test_keys_expire(redis_client: Redis) -> None:
    await redis_client.set("temporary", "x", px=80)
    assert await redis_client.get("temporary") == "x"

    await asyncio.sleep(0.2)

    assert await redis_client.get("temporary") is None


async def test_lua_scripts_run(redis_client: Redis) -> None:
    result = await redis_client.eval("return ARGV[1]", 0, "from-lua")

    assert result == "from-lua"


GRAB = """
local used = tonumber(redis.call('GET', KEYS[1]) or '0')
if used + tonumber(ARGV[1]) > tonumber(ARGV[2]) then return 0 end
redis.call('INCRBY', KEYS[1], ARGV[1])
return 1
"""


async def test_a_lua_check_and_take_never_oversubscribes(redis_client: Redis) -> None:
    """The pattern the quota system is built on: 100 callers race for 10 units."""
    grabs = await asyncio.gather(*(redis_client.eval(GRAB, 1, "budget", 1, 10) for _ in range(100)))

    assert sum(grabs) == 10
    assert await redis_client.get("budget") == "10"


async def test_redis_ready_reports_a_live_server(redis_client: Redis) -> None:
    assert await redis_ready(redis_client) is True


# ---------- the client we build ----------


def test_client_uses_the_configured_url_and_timeouts() -> None:
    settings = Settings(
        _env_file=None, redis_url="redis://cache.example:6380/3", redis_timeout=0.25
    )

    kwargs: dict[str, Any] = build_redis_client(settings).connection_pool.connection_kwargs

    assert (kwargs["host"], kwargs["port"], kwargs["db"]) == ("cache.example", 6380, 3)
    assert kwargs["socket_timeout"] == 0.25
    assert kwargs["socket_connect_timeout"] == 0.25
    assert kwargs["decode_responses"] is True


def test_pool_is_bounded_and_blocking() -> None:
    settings = Settings(_env_file=None, redis_max_connections=7, redis_timeout=0.4)

    pool = build_redis_client(settings).connection_pool

    assert isinstance(pool, BlockingConnectionPool)
    assert pool.max_connections == 7
    assert pool.timeout == 0.4


def test_defaults_point_at_local_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLMROUTER_REDIS_URL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.redis_timeout == 1.0
    assert settings.redis_max_connections == 50


def test_url_can_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMROUTER_REDIS_URL", "redis://elsewhere:6000/1")

    assert Settings(_env_file=None).redis_url == "redis://elsewhere:6000/1"


async def test_an_unreachable_server_is_not_ready_and_fails_fast() -> None:
    settings = Settings(_env_file=None, redis_url="redis://127.0.0.1:1/0", redis_timeout=0.2)
    client = build_redis_client(settings)

    try:
        assert await asyncio.wait_for(redis_ready(client), timeout=3) is False
    finally:
        await client.aclose()


# ---------- pool behaviour (needs a real server: fakeredis has no real connection pool) ----------


def real_redis_url() -> str:
    url = os.environ.get("LLMROUTER_TEST_REDIS_URL")
    if not url:
        pytest.skip("set LLMROUTER_TEST_REDIS_URL to test pooling against a real Redis")
    return url


async def test_a_burst_larger_than_the_pool_waits_instead_of_failing() -> None:
    settings = Settings(
        _env_file=None, redis_url=real_redis_url(), redis_max_connections=3, redis_timeout=2.0
    )
    client = build_redis_client(settings)
    try:
        await client.delete("burst")

        await asyncio.gather(*(client.incr("burst") for _ in range(300)))

        assert await client.get("burst") == "300"
        await client.delete("burst")
    finally:
        await client.aclose()


async def connected_clients(url: str) -> int:
    observer = Redis.from_url(url, decode_responses=True)
    try:
        info = await observer.info("clients")
        return int(info["connected_clients"])
    finally:
        await observer.aclose()


async def test_closing_the_client_releases_its_connections() -> None:
    url = real_redis_url()
    before = await connected_clients(url)
    client = build_redis_client(Settings(_env_file=None, redis_url=url))
    await asyncio.gather(*(client.ping() for _ in range(5)))
    assert await connected_clients(url) > before  # the pool really opened connections

    await client.aclose()
    await asyncio.sleep(0.1)  # the server notices closed sockets a moment later

    assert await connected_clients(url) == before


# ---------- in the running app ----------


def test_the_app_exposes_a_redis_client(client: AuthedClient, app_: FastAPI) -> None:
    assert isinstance(app_.state.redis, Redis)


def test_readiness_includes_redis(client: AuthedClient) -> None:
    body = client.get("/ready").json()

    assert body["status"] == "ready"
    assert body["checks"]["redis"] == "ok"


def test_redis_is_closed_on_shutdown(monkeypatch: pytest.MonkeyPatch, app_: FastAPI) -> None:
    closed: list[bool] = []

    class Tracking(fakeredis.FakeAsyncRedis):
        async def aclose(self, *args: Any, **kwargs: Any) -> None:
            closed.append(True)
            await super().aclose(*args, **kwargs)

    monkeypatch.setattr(
        "app.lifespan.build_redis_client", lambda _: Tracking(decode_responses=True)
    )

    with AuthedClient(app_):
        assert closed == []

    assert closed == [True]


def broken_redis_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """An app whose Redis is configured but nothing is listening."""
    unreachable = Settings(_env_file=None, redis_url="redis://127.0.0.1:1/0", redis_timeout=0.2)
    monkeypatch.setattr(
        "app.lifespan.build_redis_client", lambda _: build_redis_client(unreachable)
    )

    def registry_factory(_: httpx.AsyncClient) -> ModelRegistry:
        return registry_with(FakeProvider())

    return create_app(Settings(_env_file=None, retry_base_delay=0.0), registry_factory)


def test_the_app_starts_and_serves_even_when_redis_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    with AuthedClient(broken_redis_app(monkeypatch)) as client:
        models = client.get("/v1/models")
        chat = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert models.status_code == 200
    assert chat.status_code == 200  # nothing depends on Redis yet; quotas decide that later


def test_readiness_flags_a_down_redis_without_hiding_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with AuthedClient(broken_redis_app(monkeypatch)) as client:
        ready = client.get("/ready")
        health = client.get("/health")

    assert ready.status_code == 503
    assert ready.json()["checks"] == {"config": "ok", "database": "ok", "redis": "failed"}
    assert health.status_code == 200  # liveness must not depend on Redis


# ---------- the compose file ----------


def test_compose_file_defines_a_pinned_redis_with_a_healthcheck() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    service = compose["services"]["redis"]

    assert service["image"].startswith("redis:")
    assert not service["image"].endswith(":latest")
    assert "6379:6379" in service["ports"]
    assert "redis-cli" in service["healthcheck"]["test"]
    assert "redis-data" in compose["volumes"]
