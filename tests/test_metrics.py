import asyncio
from typing import Any

import httpx
import pytest
from prometheus_client.parser import text_string_to_metric_families
from starlette.types import Message

from app.core.config import Settings
from app.core.tiers import Tier
from app.main import create_app
from app.metrics.middleware import MetricsMiddleware
from app.metrics.registry import ATTEMPT_OUTCOMES, Metrics, safe_name, snake
from app.providers.errors import ProviderDown, RateLimited
from app.providers.fake import FakeProvider
from app.resilience.board import BreakerBoard, BreakerSettings
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry
from app.router.router import Router
from app.schemas.chat import ChatCompletionRequest
from app.schemas.chat import Message as ChatMessage
from tests.conftest import TEST_ALIAS, AuthedClient, model_config

PRICED = {"input_cost_per_mtok": 10.0, "output_cost_per_mtok": 30.0}
BODY = {"model": TEST_ALIAS, "messages": [{"role": "user", "content": "hello there"}]}


def gateway(*deployments: tuple[FakeProvider, dict[str, Any]], **settings: Any) -> AuthedClient:
    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return ModelRegistry(
            Deployment.create(model_config(**config), provider) for provider, config in deployments
        )

    options = {"retry_base_delay": 0.0, "retry_max_attempts": 1, **settings}
    return AuthedClient(create_app(Settings(_env_file=None, **options), factory))


def single(provider: FakeProvider | None = None, **settings: Any) -> AuthedClient:
    return gateway((provider or FakeProvider(), {"name": TEST_ALIAS, **PRICED}), **settings)


Sample = tuple[str, dict[str, str], float]


def scrape(client: AuthedClient, **headers: str) -> list[Sample]:
    response = client.get("/metrics", headers=headers)
    assert response.status_code == 200, response.text
    return [
        (s.name, dict(s.labels), s.value)
        for family in text_string_to_metric_families(response.text)
        for s in family.samples
    ]


def value(samples: list[Sample], name: str, **labels: str) -> float:
    """The one sample with this name whose labels include these (0 if there is none)."""
    found = [v for n, ls, v in samples if n == name and labels.items() <= ls.items()]
    assert len(found) <= 1, f"{name}{labels} matches {len(found)} series"
    return found[0] if found else 0.0


def series(samples: list[Sample], name: str) -> list[dict[str, str]]:
    return [ls for n, ls, _ in samples if n == name]


def chat(client: AuthedClient, **extra: Any) -> httpx.Response:
    return client.post("/v1/chat/completions", json={**BODY, **extra})


# ---------- the endpoint ----------


def test_the_endpoint_serves_prometheus_text() -> None:
    with single() as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain; version=")
    assert "# TYPE llmrouter_chat_requests_total counter" in response.text


def test_every_metric_is_in_the_llmrouter_namespace() -> None:
    with single() as client:
        chat(client)
        names = {n for n, _, _ in scrape(client)}
    assert names and all(n.startswith("llmrouter_") for n in names)


def test_the_endpoint_can_be_switched_off() -> None:
    with single(metrics_enabled=False) as client:
        assert client.get("/metrics").status_code == 404


def test_the_endpoint_is_not_advertised_in_the_api_docs() -> None:
    with single() as client:
        assert "/metrics" not in client.get("/openapi.json").json()["paths"]


def test_a_token_can_be_required() -> None:
    with single(metrics_token="s3cret-scrape-token") as client:
        missing = client.get("/metrics")
        wrong = client.get("/metrics", headers={"Authorization": "Bearer nope"})
        right = client.get("/metrics", headers={"Authorization": "Bearer s3cret-scrape-token"})
        basic = client.get("/metrics", headers={"Authorization": "Basic s3cret-scrape-token"})

    assert (missing.status_code, wrong.status_code, basic.status_code) == (401, 401, 401)
    assert missing.headers["www-authenticate"] == "Bearer"
    assert right.status_code == 200
    assert "s3cret-scrape-token" not in missing.text + wrong.text


def test_a_virtual_api_key_is_not_a_metrics_token() -> None:
    with single(metrics_token="s3cret-scrape-token") as client:
        admin = client.issue_key(["admin:tenants", "admin:keys"])
        assert client.get("/metrics", headers=client.as_key(admin)).status_code == 401
        assert client.get("/metrics", headers=client.as_key(client.secret)).status_code == 401


def test_the_token_is_not_shown_when_settings_are_printed() -> None:
    settings = Settings(_env_file=None, metrics_token="s3cret-scrape-token")
    assert "s3cret-scrape-token" not in repr(settings)


def test_two_gateways_keep_separate_metrics() -> None:
    with single() as first, single() as second:
        chat(first)
        chat(first)
        assert value(scrape(first), "llmrouter_chat_requests_total", outcome="success") == 2
        assert value(scrape(second), "llmrouter_chat_requests_total", outcome="success") == 0


# ---------- HTTP traffic ----------


def test_requests_are_counted_by_route_method_and_status() -> None:
    with single() as client:
        client.get("/v1/models")
        client.get("/v1/models")
        client.get("/health")
        samples = scrape(client)

    assert value(samples, "llmrouter_http_requests_total", method="GET", route="/v1/models",
                 status="200") == 2  # fmt: skip
    assert value(samples, "llmrouter_http_requests_total", route="/health", status="200") == 1


def test_routes_are_labelled_by_template_not_by_path() -> None:
    with single() as client:
        admin = client.as_key(client.issue_key(["admin:tenants"]))
        for tenant in ("aaa", "bbb", "ccc"):
            client.get(f"/admin/tenants/{tenant}", headers=admin)
        samples = scrape(client)

    routes = {ls["route"] for ls in series(samples, "llmrouter_http_requests_total")}
    assert "/admin/tenants/{tenant_id}" in routes
    assert not routes & {"/admin/tenants/aaa", "/admin/tenants/bbb"}
    assert value(samples, "llmrouter_http_requests_total", route="/admin/tenants/{tenant_id}",
                 status="404") == 3  # fmt: skip


def test_made_up_urls_cannot_mint_new_label_values() -> None:
    with single() as client:
        for n in range(30):
            assert client.get(f"/no/such/path/{n}").status_code == 404
        samples = scrape(client)

    unmatched = [ls for ls in series(samples, "llmrouter_http_requests_total")
                 if ls["status"] == "404"]  # fmt: skip
    assert unmatched == [{"method": "GET", "route": "unmatched", "status": "404"}]
    assert value(samples, "llmrouter_http_requests_total", route="unmatched") == 30


def test_scraping_is_not_counted_as_traffic() -> None:
    with single() as client:
        for _ in range(3):
            scrape(client)
        samples = scrape(client)
    assert all(ls["route"] != "/metrics" for ls in series(samples, "llmrouter_http_requests_total"))


def test_durations_are_recorded_and_finished_requests_are_not_in_progress() -> None:
    with single() as client:
        client.get("/health")
        samples = scrape(client)

    assert value(samples, "llmrouter_http_request_duration_seconds_count", route="/health") == 1
    assert value(samples, "llmrouter_http_request_duration_seconds_sum", route="/health") > 0
    # only the scrape itself is still being handled
    assert value(samples, "llmrouter_http_requests_in_progress") == 1


def test_a_streamed_response_is_one_request() -> None:
    with single(FakeProvider(reply="one two three four")) as client:
        chat(client, stream=True)
        samples = scrape(client)
    assert value(samples, "llmrouter_http_requests_total", route="/v1/chat/completions",
                 status="200") == 1  # fmt: skip


async def test_requests_that_are_not_http_pass_straight_through() -> None:
    metrics = Metrics()
    reached: list[str] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        reached.append(scope["type"])

    async def receive() -> Message:
        return {"type": "lifespan.startup"}

    async def send(message: Message) -> None:
        return None

    await MetricsMiddleware(inner, metrics)({"type": "lifespan"}, receive, send)

    assert reached == ["lifespan"]
    assert (
        list(metrics.registry.collect()) and value_of(metrics, "llmrouter_http_requests_total") == 0
    )


def value_of(metrics: Metrics, name: str) -> float:
    return sum(s.value for f in metrics.registry.collect() for s in f.samples if s.name == name)


# ---------- chat traffic ----------


def test_completions_are_counted_with_their_tokens_cost_and_duration() -> None:
    with single(FakeProvider(prompt_tokens=100, completion_tokens=50)) as client:
        for _ in range(3):
            chat(client)
        samples = scrape(client)

    labels = {"requested_model": TEST_ALIAS, "served_model": TEST_ALIAS, "outcome": "success"}
    assert value(samples, "llmrouter_chat_requests_total", **labels) == 3
    assert value(samples, "llmrouter_tokens_total", model=TEST_ALIAS, kind="prompt") == 300
    assert value(samples, "llmrouter_tokens_total", model=TEST_ALIAS, kind="completion") == 150
    assert value(samples, "llmrouter_cost_usd_total", model=TEST_ALIAS) == pytest.approx(
        3 * (100 * 10 + 50 * 30) / 1e6
    )
    assert value(samples, "llmrouter_chat_request_duration_seconds_count", stream="false") == 3


def test_a_stream_is_counted_when_it_ends() -> None:
    with single(FakeProvider(stream_usage=True, prompt_tokens=100, completion_tokens=50)) as client:
        chat(client, stream=True)
        samples = scrape(client)

    assert value(samples, "llmrouter_chat_requests_total", outcome="success") == 1
    assert value(samples, "llmrouter_chat_request_duration_seconds_count", stream="true") == 1
    assert value(samples, "llmrouter_tokens_total", kind="completion") == 50


def test_a_failure_is_counted_under_its_error_code() -> None:
    with single(FakeProvider(fail_first=99)) as client:
        chat(client)
        samples = scrape(client)

    assert value(samples, "llmrouter_chat_requests_total", outcome="upstream_unavailable",
                 served_model="none") == 1  # fmt: skip
    assert value(samples, "llmrouter_cost_usd_total", model=TEST_ALIAS) == 0


def test_a_model_name_the_caller_made_up_cannot_mint_new_series() -> None:
    with single() as client:
        for n in range(25):
            chat(client, model=f"made-up-{n}")
        samples = scrape(client)

    rows = [ls for ls in series(samples, "llmrouter_chat_requests_total")]
    assert rows == [
        {"requested_model": "unknown", "served_model": "none", "tier": "none",
         "outcome": "model_not_found"}
    ]  # fmt: skip
    assert value(samples, "llmrouter_chat_requests_total", requested_model="unknown") == 25


def test_nothing_is_labelled_by_tenant_or_key() -> None:
    with single() as client:
        chat(client)
        samples = scrape(client)
    labels = {name for _, ls, _ in samples for name in ls}
    assert not labels & {"tenant", "tenant_id", "key", "key_id", "api_key", "user"}


def routed() -> AuthedClient:
    return gateway(
        (FakeProvider(), {"name": "cheap", "tier": Tier.SLM, **PRICED}),
        (FakeProvider(), {"name": "smart", "tier": Tier.FRONTIER, **PRICED}),
    )


def test_auto_requests_are_counted_by_the_tier_that_answered() -> None:
    with routed() as client:
        chat(client, model="auto", messages=[{"role": "user", "content": "Hi!"}])
        samples = scrape(client)

    assert value(samples, "llmrouter_auto_requests_total", tier="slm", escalated="false") == 1
    assert value(samples, "llmrouter_chat_requests_total", requested_model="auto",
                 served_model="cheap", tier="slm") == 1  # fmt: skip


def test_an_escalation_is_counted_as_one() -> None:
    with gateway(
        (FakeProvider(fail_first=99), {"name": "cheap", "tier": Tier.SLM, **PRICED}),
        (FakeProvider(), {"name": "smart", "tier": Tier.FRONTIER, **PRICED}),
    ) as client:
        chat(client, model="auto", messages=[{"role": "user", "content": "Hi!"}])
        samples = scrape(client)

    assert value(samples, "llmrouter_auto_requests_total", tier="frontier", escalated="true") == 1


def test_a_plain_request_is_not_an_auto_request() -> None:
    with routed() as client:
        chat(client, model="cheap")
        samples = scrape(client)
    assert series(samples, "llmrouter_auto_requests_total") == []


def test_metrics_do_not_depend_on_the_request_log() -> None:
    with single(request_log_enabled=False) as client:
        chat(client)
        samples = scrape(client)
    assert value(samples, "llmrouter_chat_requests_total", outcome="success") == 1


def test_a_failing_metrics_observer_does_not_fail_the_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken(*args: Any) -> None:
        raise RuntimeError("metrics bug")

    with single() as client:
        client.app.state.request_logger._sinks[:] = [broken]  # type: ignore[attr-defined]
        with caplog.at_level("ERROR", logger="app.usage.record"):
            response = chat(client)

    assert response.status_code == 200
    assert any("a request observer failed" in r.message for r in caplog.records)


# ---------- upstream health ----------


def test_every_attempt_on_a_deployment_is_counted_by_how_it_ended() -> None:
    with single(FakeProvider(fail_first=1), retry_max_attempts=3) as client:
        assert chat(client).status_code == 200  # the first attempt fails, the retry succeeds
        samples = scrape(client)

    assert value(samples, "llmrouter_upstream_attempts_total", model=TEST_ALIAS,
                 outcome="provider_down") == 1  # fmt: skip
    assert value(samples, "llmrouter_upstream_attempts_total", model=TEST_ALIAS,
                 outcome="success") == 1  # fmt: skip
    assert value(samples, "llmrouter_upstream_attempt_duration_seconds_count",
                 model=TEST_ALIAS) == 1  # fmt: skip


def test_a_surprising_error_class_cannot_mint_a_new_label_value() -> None:
    metrics = Metrics()
    metrics.set_models(["m"])
    deployment = Deployment.create(model_config(name="m"), FakeProvider())

    for name in ("SomethingNobodyExpected", "AnotherOne", "ProviderDown"):
        metrics.observe_attempt(deployment, name, 0.1)

    assert value_of(metrics, "llmrouter_upstream_attempts_total") == 3
    outcomes = {s.labels["outcome"] for f in metrics.registry.collect() for s in f.samples
                if s.name == "llmrouter_upstream_attempts_total"}  # fmt: skip
    assert outcomes == {"other", "provider_down"}


def test_outcome_names_become_snake_case_labels() -> None:
    assert snake("ProviderDown") == "provider_down"
    assert snake("ContextLengthExceededError") == "context_length_exceeded_error"
    assert {snake(o) for o in ATTEMPT_OUTCOMES} >= {"success", "refused", "rate_limited"}


async def test_a_refused_attempt_is_counted_without_a_call() -> None:
    seen: list[tuple[str, str]] = []
    board = BreakerBoard(BreakerSettings(failure_threshold=1, failure_rate_threshold=None))
    provider = FakeProvider()
    target = Deployment.create(model_config(name="m"), provider)
    router = Router(
        ModelRegistry([target]),
        breakers=board,
        on_attempt=lambda d, o, s: seen.append((d.alias, o)),
    )
    board.acquire(target).failure(ProviderDown("x"))  # opens the breakers

    from app.resilience.board import DeploymentUnavailable

    with pytest.raises(DeploymentUnavailable):
        await router._guarded(target, lambda: asyncio.sleep(0))

    assert seen == [("m", "refused")] and provider.calls == 0


async def test_a_failing_attempt_observer_does_not_fail_the_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken(*args: Any) -> None:
        raise RuntimeError("observer bug")

    router = Router(
        ModelRegistry([Deployment.create(model_config(name=TEST_ALIAS), FakeProvider())]),
        on_attempt=broken,
    )
    request = ChatCompletionRequest(
        model=TEST_ALIAS, messages=[ChatMessage(role="user", content="hi")]
    )
    with caplog.at_level("ERROR"):
        response = await router.complete(request)

    assert response.choices
    assert any("an attempt observer failed" in r.message for r in caplog.records)


def test_circuit_breaker_transitions_are_counted_by_kind() -> None:
    with single(FakeProvider(fail_first=99), breaker_failure_threshold=1) as client:
        chat(client)
        samples = scrape(client)

    assert value(samples, "llmrouter_circuit_breaker_transitions_total", kind="deployment",
                 to_state="open") == 1  # fmt: skip
    assert value(samples, "llmrouter_circuit_breaker_transitions_total", kind="provider",
                 to_state="open") == 1  # fmt: skip


def test_an_open_breaker_shows_in_the_live_gauge() -> None:
    with single(FakeProvider(fail_first=99), breaker_failure_threshold=1) as client:
        chat(client)
        samples = scrape(client)

    states = {ls["kind"]: v for n, ls, v in samples if n == "llmrouter_circuit_breaker_state"}
    assert states == {"deployment": 2.0, "provider": 2.0}


def test_a_healthy_breaker_reads_closed() -> None:
    with single() as client:
        chat(client)
        samples = scrape(client)
    states = [v for n, _, v in samples if n == "llmrouter_circuit_breaker_state"]
    assert states == [0.0, 0.0]


def test_a_429_is_counted_and_the_rest_is_visible() -> None:
    provider = FakeProvider(fail_first=99, error_factory=lambda: RateLimited("x", retry_after=30))
    with single(provider) as client:
        chat(client)
        samples = scrape(client)

    assert value(samples, "llmrouter_upstream_throttles_total", model=TEST_ALIAS) == 1
    resting = value(samples, "llmrouter_deployment_throttled_seconds", model=TEST_ALIAS)
    assert 0 < resting <= 30


def test_the_latency_the_router_has_measured_is_exposed() -> None:
    with single() as client:
        chat(client)
        samples = scrape(client)
    assert value(samples, "llmrouter_deployment_latency_seconds", model=TEST_ALIAS) > 0


def test_a_deployment_never_called_has_no_latency_series() -> None:
    with single() as client:
        samples = scrape(client)
    assert series(samples, "llmrouter_deployment_latency_seconds") == []


# ---------- labels that could carry secrets ----------


def test_credentials_in_a_provider_url_never_reach_a_label() -> None:
    assert (
        safe_name("openai_compatible@https://user:pw@api.example.com:8443/v1?key=abc")
        == "openai_compatible@https://api.example.com:8443/v1"
    )
    assert safe_name("deployment:fast:dep_123") == "deployment:fast:dep_123"


def test_the_live_gauge_labels_carry_no_secrets() -> None:
    config = {"name": TEST_ALIAS, "base_url": "https://user:pw@api.example.com/v1?key=abc"}
    with gateway((FakeProvider(), config)) as client:
        chat(client)
        text = client.get("/metrics").text
    assert "pw@" not in text and "key=abc" not in text and "user:" not in text


# ---------- the app can be started twice ----------


def test_starting_the_same_app_twice_does_not_double_register_the_live_gauges() -> None:
    client = single()
    with client:
        pass
    with client:  # the lifespan runs again against the same Metrics object
        chat(client)
        samples = scrape(client)
    assert value(samples, "llmrouter_chat_requests_total", outcome="success") == 1
    assert len([n for n, _, _ in samples if n == "llmrouter_circuit_breaker_state"]) == 2


def test_unknown_models_fold_into_unknown_and_none_means_none() -> None:
    metrics = Metrics()
    metrics.set_models(["fast", "smart"])
    assert [metrics.model(m) for m in ("fast", "smart", "auto", "other", None)] == [
        "fast", "smart", "auto", "unknown", "none",
    ]  # fmt: skip
