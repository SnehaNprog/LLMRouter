"""When the gateway runs out of connections to a provider, that is the gateway's problem.

Found by the load test: with a healthy but slow provider and more requests in flight than the
connection pool holds, the ones that waited too long for a connection were counted as the
provider timing out. That opened the circuit breaker, which then refused every request for as long
as it stayed open, though the provider had not failed once.
"""

import asyncio
from collections.abc import Iterator

import httpx
import pytest

from app.api.errors import outcome_of
from app.core.config import Settings
from app.loadtest.harness import Stack, StackConfig, running_stack
from app.loadtest.stub import StubSettings
from app.main import create_app
from app.providers.errors import GatewayOverloaded, ProviderDown, Timeout
from app.providers.fake import FakeProvider
from app.providers.http_errors import translate_transport_errors
from app.providers.openai_compat import OpenAICompatibleProvider
from app.resilience.board import BreakerBoard, BreakerSettings, Health, health_of
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry
from app.router.retry import Decision, classify
from app.router.router import Router
from app.schemas.chat import ChatCompletionRequest, Message
from tests.conftest import TEST_ALIAS, AuthedClient, model_config

BODY = {"model": TEST_ALIAS, "messages": [{"role": "user", "content": "hello there"}]}


# ---------- translating the error ----------


async def test_a_pool_timeout_is_the_gateways_problem_not_the_vendors() -> None:
    with pytest.raises(GatewayOverloaded) as caught:
        async with translate_transport_errors("vendor"):
            raise httpx.PoolTimeout("no free connection")
    assert caught.value.provider == "vendor"
    assert isinstance(caught.value.__cause__, httpx.PoolTimeout)


@pytest.mark.parametrize("error", [httpx.ReadTimeout("slow"), httpx.ConnectTimeout("slow")])
async def test_other_timeouts_are_still_the_vendors(error: Exception) -> None:
    with pytest.raises(Timeout):
        async with translate_transport_errors("vendor"):
            raise error


async def test_the_real_adapter_raises_it_when_the_pool_is_exhausted() -> None:
    def pool_is_full(request: httpx.Request) -> httpx.Response:
        raise httpx.PoolTimeout("no free connection")

    async with httpx.AsyncClient(transport=httpx.MockTransport(pool_is_full)) as client:
        provider = OpenAICompatibleProvider(
            "v", base_url="http://v/v1", api_key=None, client=client
        )
        request = ChatCompletionRequest(model="m", messages=[Message(role="user", content="hi")])
        with pytest.raises(GatewayOverloaded):
            await provider.chat(request)
        with pytest.raises(GatewayOverloaded):
            [chunk async for chunk in provider.stream(request)]


# ---------- what is done about it ----------


def test_it_does_not_count_against_any_circuit_breaker() -> None:
    assert health_of(GatewayOverloaded("x")) is Health.NEUTRAL
    assert health_of(Timeout("x")) is Health.PROVIDER_DOWN  # while a real timeout still does

    board = BreakerBoard(BreakerSettings(failure_threshold=1, failure_rate_threshold=None))
    target = Deployment.create(model_config(name="m"), FakeProvider())
    for _ in range(50):
        board.acquire(target).failure(GatewayOverloaded("x"))
    assert not board.is_open(target)


def test_it_is_neither_retried_nor_sent_to_a_fallback() -> None:
    assert classify(GatewayOverloaded("x")) is Decision.FAIL


async def test_the_router_gives_up_at_once_and_does_not_try_another_model() -> None:
    overloaded = FakeProvider(fail_first=99, error_factory=lambda: GatewayOverloaded("x"))
    fallback = FakeProvider()
    router = Router(
        ModelRegistry(
            [
                Deployment.create(model_config(name="a"), overloaded),
                Deployment.create(model_config(name="b"), fallback),
            ],
            {"a": ["b"]},
        )
    )
    request = ChatCompletionRequest(model="a", messages=[Message(role="user", content="hi")])

    with pytest.raises(GatewayOverloaded):
        await router.complete(request)

    assert overloaded.calls == 1 and fallback.calls == 0


def test_the_client_is_told_to_come_back_in_a_moment() -> None:
    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        provider = FakeProvider(fail_first=99, error_factory=lambda: GatewayOverloaded("x"))
        return ModelRegistry([Deployment.create(model_config(), provider)])

    settings = Settings(_env_file=None, retry_base_delay=0.0)
    with AuthedClient(create_app(settings, factory)) as client:
        response = client.post("/v1/chat/completions", json=BODY)
        metrics = client.get("/metrics").text

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "gateway_overloaded"
    assert response.headers["retry-after"] == "1"
    # counted as what it was, not folded into "other"
    attempts = [
        line
        for line in metrics.splitlines()
        if line.startswith("llmrouter_upstream_attempts_total")
    ]
    assert len(attempts) == 1 and 'outcome="gateway_overloaded"' in attempts[0]


def test_it_is_logged_under_its_own_code() -> None:
    assert outcome_of(GatewayOverloaded("x")) == (503, "gateway_overloaded")
    assert outcome_of(ProviderDown("x")) == (502, "upstream_unavailable")


# ---------- the incident, reproduced ----------


@pytest.fixture(scope="module")
def tiny_pool_stack() -> Iterator[Stack]:
    """A gateway that may hold only two connections to the provider and waits 0.2 s for one."""
    config = StackConfig(
        stub=StubSettings(latency_ms=10),
        gateway_env={
            "LLMROUTER_HTTP_MAX_CONNECTIONS": "2",
            "LLMROUTER_HTTP_MAX_KEEPALIVE_CONNECTIONS": "2",
            "LLMROUTER_HTTP_POOL_TIMEOUT": "0.2",
            "LLMROUTER_RETRY_MAX_ATTEMPTS": "1",
            "LLMROUTER_BREAKER_FAILURE_THRESHOLD": "3",
            "LLMROUTER_BREAKER_COOLDOWN": "30",
            "LLMROUTER_BREAKER_SHARED": "false",
        },
    )
    with running_stack(config) as running:
        yield running


def test_a_burst_beyond_the_pool_is_shed_but_a_healthy_provider_stays_in_service(
    tiny_pool_stack: Stack,
) -> None:
    stack = tiny_pool_stack
    key = stack.make_key()
    stack.stub_configure(latency_ms=1500)  # every answer now takes 1.5 s, so the pool stays full

    async def burst() -> list[httpx.Response]:
        async with httpx.AsyncClient(base_url=stack.gateway_url, timeout=30) as client:
            body = {"model": "small", "messages": [{"role": "user", "content": "hi"}]}
            headers = {"Authorization": f"Bearer {key}"}
            return list(
                await asyncio.gather(
                    *[
                        client.post("/v1/chat/completions", json=body, headers=headers)
                        for _ in range(12)
                    ]
                )
            )

    responses = asyncio.run(burst())
    codes = sorted(r.status_code for r in responses)
    stack.stub_configure(latency_ms=10)

    assert codes.count(200) == 2  # as many as the pool could carry
    assert codes.count(503) == 10
    assert {r.json()["error"]["code"] for r in responses if r.status_code == 503} == {
        "gateway_overloaded"
    }
    assert stack.stub_stats()["errors"] == 0  # the provider never failed

    # and it is still usable straight away: no breaker opened on its account
    async def afterwards() -> int:
        async with httpx.AsyncClient(base_url=stack.gateway_url, timeout=30) as client:
            body = {"model": "small", "messages": [{"role": "user", "content": "hi"}]}
            response = await client.post(
                "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}
            )
            return response.status_code

    assert asyncio.run(afterwards()) == 200


def test_the_settings_it_relies_on_exist() -> None:
    names = set(Settings.model_fields)
    assert {"http_max_connections", "http_max_keepalive_connections", "http_pool_timeout"} <= names
    assert isinstance(Settings(_env_file=None).http_max_connections, int)
