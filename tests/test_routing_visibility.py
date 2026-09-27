import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest
from openai import OpenAI

from app.core.config import Settings
from app.core.logging import JsonFormatter
from app.core.tiers import Tier
from app.main import create_app
from app.providers.fake import FakeProvider
from app.router.deployment import Deployment
from app.router.latency import LatencyTracker
from app.router.registry import ModelRegistry
from app.routing.explain import routing_headers
from app.routing.score import SIGNALS, score_request
from app.routing.selector import choose_model
from app.schemas.chat import ChatCompletionRequest
from tests.conftest import AuthedClient, model_config
from tests.test_auto_routing import EASY, HARD, MEDIUM, answer, ask
from tests.test_escalation import dep, tiered


def route_headers(response: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in response.headers.items() if k.startswith("x-llmrouter-")}


def request_of(text: str) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {"model": "auto", "messages": [{"role": "user", "content": text}]}
    )


# ---------- the response headers ----------


def test_an_easy_request_is_explained() -> None:
    client, _ = tiered()
    with client:
        response = ask(client, None, EASY)

    expected = score_request(request_of(EASY))
    assert route_headers(response) == {
        "x-llmrouter-model": "cheap",
        "x-llmrouter-tier": "slm",
        "x-llmrouter-required-tier": "slm",
        "x-llmrouter-complexity": f"{expected.value:.2f}",
        "x-llmrouter-reason": ",".join(expected.top(3)),
        "x-llmrouter-escalated": "false",
    }


def test_a_hard_request_names_the_signals_that_made_it_hard() -> None:
    client, _ = tiered()
    with client:
        headers = route_headers(ask(client, None, HARD))

    assert headers["x-llmrouter-model"] == "top"
    assert headers["x-llmrouter-tier"] == headers["x-llmrouter-required-tier"] == "frontier"
    assert headers["x-llmrouter-reason"].split(",")[0] == "reasoning"
    assert float(headers["x-llmrouter-complexity"]) > 0.65


@pytest.mark.parametrize("prompt", [EASY, MEDIUM, HARD])
def test_the_reported_complexity_is_the_real_score(prompt: str) -> None:
    client, _ = tiered()
    with client:
        header = route_headers(ask(client, None, prompt))["x-llmrouter-complexity"]

    assert header == f"{score_request(request_of(prompt)).value:.2f}"


def test_an_escalated_request_says_who_answered_and_that_it_escalated() -> None:
    client, _ = tiered(fail={"cheap": 99})
    with client:
        response = ask(client, None, EASY)

    headers = route_headers(response)
    assert answer(response) == "answer from middle"
    assert headers["x-llmrouter-model"] == "middle"  # who actually answered
    assert headers["x-llmrouter-tier"] == "mid"
    assert headers["x-llmrouter-required-tier"] == "slm"  # what the request needed
    assert headers["x-llmrouter-escalated"] == "true"


def test_a_request_nothing_could_fully_serve_is_flagged_degraded() -> None:
    only_small = ModelRegistry([dep("small", FakeProvider("small"), Tier.SLM)])
    app = create_app(Settings(_env_file=None, retry_base_delay=0.0), lambda _: only_small)
    with AuthedClient(app) as client:
        hard = route_headers(ask(client, None, HARD))
        easy = route_headers(ask(client, None, EASY))

    assert hard["x-llmrouter-degraded"] == "true"  # the best available, not what it needed
    assert "x-llmrouter-degraded" not in easy


def test_streams_carry_the_headers_too_even_after_an_escalation() -> None:
    client, _ = tiered(fail={"cheap": 99})
    with client:
        response = ask(client, None, EASY, stream=True)

    assert response.status_code == 200
    headers = route_headers(response)
    assert headers["x-llmrouter-model"] == "middle"
    assert headers["x-llmrouter-escalated"] == "true"


def test_requests_for_a_named_model_get_no_routing_headers() -> None:
    client, _ = tiered()
    with client:
        response = ask(client, None, EASY, model="cheap")

    assert route_headers(response) == {}


def test_a_fallback_never_reveals_a_model_the_key_may_not_know() -> None:
    client, _ = tiered(fail={"cheap": 99}, fallbacks={"cheap": ["top"]})
    with client:
        secret = client.issue_key(allowed_models=["cheap"])

        response = ask(client, secret, EASY, model="cheap")

    assert response.status_code == 200  # served by the fallback ...
    assert route_headers(response) == {}  # ... without saying which model that was
    assert "top" not in str(dict(response.headers))


def test_the_headers_can_be_switched_off() -> None:
    client, _ = tiered(routing_headers=False)
    with client:
        response = ask(client, None, EASY)

    assert response.status_code == 200 and route_headers(response) == {}


def test_failures_do_not_carry_routing_headers() -> None:
    client, _ = tiered(fail={"cheap": 99, "middle": 99, "top": 99})
    with client:
        down = ask(client, None, EASY)
        forbidden = ask(client, client.issue_key(allowed_models=["cheap"]), EASY)
        limited_key = client.issue_key(rpm_limit=1)
        ask(client, limited_key, EASY)
        limited = ask(client, limited_key, EASY)

    for response, status in ((down, 502), (forbidden, 403), (limited, 429)):
        assert response.status_code == status
        assert route_headers(response) == {}


def test_the_prompt_never_appears_in_any_header() -> None:
    secret_text = "the launch codes are 1234-SECRET-5678"
    client, _ = tiered()
    with client:
        response = ask(client, None, secret_text + " Compare the options and justify.")

    assert "SECRET" not in str(dict(response.headers)) and "1234" not in str(dict(response.headers))


def test_the_reasons_come_only_from_the_known_signals() -> None:
    client, _ = tiered()
    with client:
        reasons = [
            route_headers(ask(client, None, p))["x-llmrouter-reason"].split(",")
            for p in (EASY, MEDIUM, HARD, "hi", "Prove it: $x^2 = 4$, step by step.")
        ]

    for names in reasons:
        assert all(name in SIGNALS or name == "none" for name in names)


def test_the_sdk_can_read_the_explanation() -> None:
    client, _ = tiered()
    with client:
        sdk = OpenAI(base_url="http://testserver/v1", api_key=client.secret, http_client=client)
        raw: Any = sdk.chat.completions.with_raw_response.create(
            model="auto", messages=[{"role": "user", "content": HARD}]
        )

    assert raw.headers["x-llmrouter-model"] == "top"
    assert raw.parse().choices[0].message.content == "answer from top"


def test_concurrent_requests_each_get_their_own_explanation() -> None:
    client, _ = tiered()
    expected = {EASY: "cheap", MEDIUM: "middle", HARD: "top"}
    with client:

        def call(index: int) -> tuple[str, str, str]:
            text = (EASY, MEDIUM, HARD)[index % 3]
            response = ask(client, None, text)
            return text, route_headers(response)["x-llmrouter-model"], answer(response)

        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(call, range(48)))

    for text, header_model, content in results:
        assert header_model == expected[text]
        assert content == f"answer from {expected[text]}"


# ---------- header safety ----------


def test_unusual_model_names_are_made_safe_for_a_header() -> None:
    nasty = "my model\r\nX-Injected: yes/é£" + "x" * 300
    deployment = Deployment.create(
        model_config(name=nasty, model="m", tier=Tier.SLM), FakeProvider("n")
    )
    registry = ModelRegistry([deployment])
    choice = choose_model(request_of("hi"), registry, LatencyTracker())

    headers = routing_headers(choice, deployment)

    value = headers["x-llmrouter-model"]
    assert "\r" not in value and "\n" not in value and " " not in value
    assert value.isascii() and len(value) <= 100
    assert value.startswith("my_model__X-Injected:_yes/")
    assert all(k.isascii() and v.isascii() and "\n" not in v for k, v in headers.items())


# ---------- the log ----------


def routing_records(caplog: pytest.LogCaptureFixture, message: str) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records if r.name == "app.routing.decisions" and r.getMessage() == message
    ]


def test_each_decision_is_logged_with_its_reasoning(caplog: pytest.LogCaptureFixture) -> None:
    client, _ = tiered()
    with client, caplog.at_level(logging.INFO):
        ask(client, None, HARD)

    (record,) = routing_records(caplog, "routing decision")
    expected = score_request(request_of(HARD))
    assert record.requested_model == "auto"  # type: ignore[attr-defined]
    assert record.chosen == "top"  # type: ignore[attr-defined]
    assert record.tier == "frontier" and record.required_tier == "frontier"  # type: ignore[attr-defined]
    assert record.clears_bar is True  # type: ignore[attr-defined]
    assert record.complexity == expected.value  # type: ignore[attr-defined]
    assert set(record.signals) <= set(SIGNALS)  # type: ignore[attr-defined]
    assert list(record.signals)[0] == "reasoning"  # type: ignore[attr-defined]
    assert record.tenant_id and record.key_id  # type: ignore[attr-defined]
    candidate = record.candidates[0]  # type: ignore[attr-defined]
    assert set(candidate) == {
        "alias",
        "tier",
        "expected_cost_usd",
        "expected_latency_s",
        "latency_measured",
        "utility",
    }


def test_the_outcome_is_logged_with_who_answered(caplog: pytest.LogCaptureFixture) -> None:
    client, _ = tiered()
    with client, caplog.at_level(logging.INFO):
        ask(client, None, EASY)

    (record,) = routing_records(caplog, "request routed")
    assert (record.chosen, record.served, record.served_tier) == ("cheap", "cheap", "slm")  # type: ignore[attr-defined]
    assert record.escalated is False  # type: ignore[attr-defined]


def test_an_escalation_is_visible_in_the_log_in_order(caplog: pytest.LogCaptureFixture) -> None:
    client, _ = tiered(fail={"cheap": 99})
    with client, caplog.at_level(logging.INFO):
        ask(client, None, EASY)

    story = [
        r.getMessage()
        for r in caplog.records
        if r.name in ("app.routing.decisions", "app.router.router")
        and r.getMessage()
        in ("routing decision", "model exhausted, falling back", "request routed")
    ]
    assert story == ["routing decision", "model exhausted, falling back", "request routed"]
    (outcome,) = routing_records(caplog, "request routed")
    assert (outcome.chosen, outcome.served, outcome.escalated) == ("cheap", "middle", True)  # type: ignore[attr-defined]


def test_named_models_are_not_routing_decisions(caplog: pytest.LogCaptureFixture) -> None:
    client, _ = tiered()
    with client, caplog.at_level(logging.INFO):
        ask(client, None, EASY, model="cheap")

    assert [r for r in caplog.records if r.name == "app.routing.decisions"] == []


def test_the_log_is_written_even_when_the_headers_are_off(caplog: pytest.LogCaptureFixture) -> None:
    client, _ = tiered(routing_headers=False)
    with client, caplog.at_level(logging.INFO):
        ask(client, None, EASY)

    assert len(routing_records(caplog, "routing decision")) == 1
    assert len(routing_records(caplog, "request routed")) == 1


def test_a_failed_request_still_leaves_its_decision_in_the_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, _ = tiered(fail={"cheap": 99, "middle": 99, "top": 99})
    with client, caplog.at_level(logging.INFO):
        response = ask(client, None, EASY)

    assert response.status_code == 502
    assert len(routing_records(caplog, "routing decision")) == 1  # explains what was attempted
    assert routing_records(caplog, "request routed") == []  # nothing answered


def test_the_prompt_is_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    secret_text = "the launch codes are 1234-SECRET-5678"
    client, _ = tiered()
    with client, caplog.at_level(logging.DEBUG):
        ask(client, None, secret_text + ". Compare the trade-offs and justify the answer.")

    everything = " ".join(
        f"{r.getMessage()} {r.__dict__}" for r in caplog.records if r.name.startswith("app.rout")
    )
    assert "SECRET" not in everything and "launch codes" not in everything


def test_only_the_best_few_candidates_are_logged_in_detail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    spec: dict[str, tuple[Tier, float, float, int | None]] = {
        f"m{n}": (Tier.SLM, 0.05 + n / 100, 0.08, None) for n in range(9)
    }
    client, _ = tiered_many(spec)
    with client, caplog.at_level(logging.INFO):
        ask(client, None, EASY)

    (record,) = routing_records(caplog, "routing decision")
    candidates = record.candidates  # type: ignore[attr-defined]
    assert len(candidates) == 5
    assert [c["alias"] for c in candidates] == ["m0", "m1", "m2", "m3", "m4"]  # best first


def tiered_many(spec: dict[str, tuple[Tier, float, float, int | None]]) -> tuple[AuthedClient, Any]:
    providers = {name: FakeProvider(name, reply=f"answer from {name}") for name in spec}

    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return ModelRegistry(
            [
                dep(name, providers[name], tier, input_cost_per_mtok=a, output_cost_per_mtok=b)
                for name, (tier, a, b, _) in spec.items()
            ]
        )

    return AuthedClient(
        create_app(Settings(_env_file=None, retry_base_delay=0.0), factory)
    ), providers


def test_log_lines_are_valid_json_that_carry_the_request_id() -> None:
    client, _ = tiered()
    lines: list[dict[str, Any]] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(json.loads(JsonFormatter().format(record)))

    logger = logging.getLogger("app.routing.decisions")
    handler = Capture()
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.INFO)
    try:
        with client:
            client.post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": EASY}]},
                headers={**client.as_key(client.secret), "X-Request-ID": "trace-xyz"},
            )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)

    assert [line["message"] for line in lines] == ["routing decision", "request routed"]
    assert all(line["request_id"] == "trace-xyz" for line in lines)
    assert lines[0]["candidates"][0]["alias"] == "cheap"


def test_concurrent_requests_leave_logs_that_match_their_own_responses() -> None:
    client, _ = tiered()
    lines: list[dict[str, Any]] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(json.loads(JsonFormatter().format(record)))

    logger = logging.getLogger("app.routing.decisions")
    handler = Capture()
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.INFO)
    prompts = (EASY, MEDIUM, HARD)
    try:
        with client:

            def call(index: int) -> tuple[str, str]:
                request_id = f"req-{index}"
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "auto",
                        "messages": [{"role": "user", "content": prompts[index % 3]}],
                    },
                    headers={**client.as_key(client.secret), "X-Request-ID": request_id},
                )
                return request_id, response.headers["x-llmrouter-model"]

            with ThreadPoolExecutor(max_workers=12) as pool:
                answered = dict(pool.map(call, range(36)))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)

    logged = {
        line["request_id"]: line["served"] for line in lines if line["message"] == "request routed"
    }
    assert logged == answered  # every log line belongs to the request that produced it
    assert len(lines) == 72  # a decision and an outcome for each of the 36


def test_the_header_and_the_log_agree(caplog: pytest.LogCaptureFixture) -> None:
    client, _ = tiered(fail={"cheap": 99})
    with client, caplog.at_level(logging.INFO):
        response = ask(client, None, EASY)

    (decision,) = routing_records(caplog, "routing decision")
    (outcome,) = routing_records(caplog, "request routed")
    headers = route_headers(response)
    assert headers["x-llmrouter-complexity"] == f"{decision.complexity:.2f}"  # type: ignore[attr-defined]
    assert headers["x-llmrouter-model"] == outcome.served  # type: ignore[attr-defined]
    assert headers["x-llmrouter-required-tier"] == decision.required_tier  # type: ignore[attr-defined]
    assert (headers["x-llmrouter-escalated"] == "true") == outcome.escalated  # type: ignore[attr-defined]


def test_the_default_is_to_explain_and_the_setting_can_be_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLMROUTER_ROUTING_HEADERS", raising=False)
    assert Settings(_env_file=None).routing_headers is True

    monkeypatch.setenv("LLMROUTER_ROUTING_HEADERS", "false")
    assert Settings(_env_file=None).routing_headers is False


def test_headers_use_lowercase_names_a_client_can_rely_on() -> None:
    client, _ = tiered()
    with client:
        response = ask(client, None, EASY)

    assert all(re.fullmatch(r"x-llmrouter-[a-z-]+", name) for name in route_headers(response))
