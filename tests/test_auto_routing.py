from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import OpenAI

from app.core.config import Settings
from app.core.model_config import ConfigError, parse_config
from app.core.tiers import Tier
from app.main import create_app
from app.providers.fake import FakeProvider
from app.router.context import ContextLengthExceededError
from app.router.deployment import Deployment
from app.router.latency import LatencyTracker
from app.router.registry import ModelRegistry, UnknownModelError
from app.routing.policy import AUTO_MODEL, RoutingPolicy
from app.routing.resolve import resolve_model
from app.schemas.chat import ChatCompletionRequest
from tests.conftest import AuthedClient, model_config

EASY = "What is the capital of France?"
MEDIUM = "Write a Python function that checks if a number is prime."
HARD = (
    "Design a distributed rate limiter. Compare token bucket vs sliding window, analyze the "
    "trade-offs step by step, and respond in JSON. It must not lose requests.\n"
    "1. Data model\n2. Failure modes\n3. Edge cases"
)


@dataclass
class World:
    client: AuthedClient
    providers: dict[str, FakeProvider]

    def served_by(self) -> list[str]:
        return [name for name, provider in self.providers.items() if provider.calls]


def build(
    *,
    tiers: dict[str, tuple[Tier | None, float, float, int | None]] | None = None,
    fallbacks: dict[str, list[str]] | None = None,
    policy: RoutingPolicy | None = None,
    **settings: Any,
) -> World:
    spec = tiers or {
        "cheap": (Tier.SLM, 0.05, 0.08, 131_072),
        "middle": (Tier.MID, 0.6, 0.8, 131_072),
        "top": (Tier.FRONTIER, 3.0, 15.0, 200_000),
    }
    providers = {name: FakeProvider(name, reply=f"answer from {name}") for name in spec}

    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return ModelRegistry(
            [
                Deployment.create(
                    model_config(
                        name=name,
                        model=f"{name}-up",
                        tier=tier,
                        input_cost_per_mtok=cost_in,
                        output_cost_per_mtok=cost_out,
                        context_window=window,
                    ),
                    providers[name],
                )
                for name, (tier, cost_in, cost_out, window) in spec.items()
            ],
            fallbacks,
            routing_policy=policy,
        )

    options = {"retry_base_delay": 0.0, "retry_max_attempts": 1, **settings}
    return World(AuthedClient(create_app(Settings(_env_file=None, **options), factory)), providers)


def ask(
    client: AuthedClient, secret: str | None, content: str, model: str = AUTO_MODEL, **extra: Any
) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": content}], **extra},
        headers=client.as_key(secret or client.secret),
    )


def answer(response: httpx.Response) -> str:
    body: dict[str, Any] = response.json()
    return str(body["choices"][0]["message"]["content"])


# ---------- the routing itself ----------


def test_an_easy_request_is_answered_by_the_small_model() -> None:
    world = build()
    with world.client as client:
        response = ask(client, None, EASY)

    assert response.status_code == 200
    assert answer(response) == "answer from cheap"
    assert world.served_by() == ["cheap"]


def test_a_moderate_request_is_answered_by_the_mid_tier_model() -> None:
    world = build()
    with world.client as client:
        response = ask(client, None, MEDIUM)

    assert answer(response) == "answer from middle"
    assert world.served_by() == ["middle"]


def test_a_hard_request_is_answered_by_the_frontier_model() -> None:
    world = build()
    with world.client as client:
        response = ask(client, None, HARD)

    assert answer(response) == "answer from top"
    assert world.served_by() == ["top"]


def test_the_chosen_models_own_upstream_name_is_sent_to_its_provider() -> None:
    world = build()
    with world.client as client:
        response = ask(client, None, EASY)

    assert world.providers["cheap"].requests[0].model == "cheap-up"
    assert response.json()["model"] == "cheap-up"  # never the literal "auto"


def test_streaming_requests_are_routed_too() -> None:
    world = build()
    with world.client as client:
        response = ask(client, None, HARD, stream=True)

    assert response.status_code == 200
    assert "answer" in response.text and "from" in response.text
    assert world.served_by() == ["top"]


def test_asking_for_a_model_by_name_still_works_and_is_not_rerouted() -> None:
    world = build()
    with world.client as client:
        response = ask(client, None, EASY, model="top")  # an easy question, but they asked for top

    assert answer(response) == "answer from top"
    assert world.served_by() == ["top"]


def test_each_request_is_judged_on_its_own_merits() -> None:
    world = build()
    with world.client as client:
        replies = [answer(ask(client, None, text)) for text in (EASY, HARD, MEDIUM, EASY)]

    assert replies == [
        "answer from cheap",
        "answer from top",
        "answer from middle",
        "answer from cheap",
    ]


def test_concurrent_requests_each_reach_the_right_model() -> None:
    world = build()
    prompts = [EASY, MEDIUM, HARD] * 16
    expected = {EASY: "cheap", MEDIUM: "middle", HARD: "top"}
    with world.client as client:

        def call(text: str) -> tuple[str, str]:
            return text, answer(ask(client, None, text))

        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(call, prompts))

    assert all(reply == f"answer from {expected[text]}" for text, reply in results)
    assert {name: p.calls for name, p in world.providers.items()} == {
        "cheap": 16,
        "middle": 16,
        "top": 16,
    }


def test_the_sdk_can_use_auto() -> None:
    world = build()
    with world.client as client:
        sdk = OpenAI(base_url="http://testserver/v1", api_key=client.secret, http_client=client)

        completion = sdk.chat.completions.create(
            model="auto", messages=[{"role": "user", "content": EASY}]
        )
        listed = [m.id for m in sdk.models.list().data]

    assert completion.choices[0].message.content == "answer from cheap"
    assert listed[0] == "auto"


# ---------- tuning the routing ----------


def test_the_tier_ceilings_move_the_boundaries() -> None:
    cautious = RoutingPolicy.from_overrides({"slm_ceiling": 0.05, "mid_ceiling": 0.1})
    world = build(policy=cautious)
    with world.client as client:
        response = ask(client, None, EASY)  # fine by default, but these ceilings distrust it

    assert answer(response) == "answer from top"


def test_measured_latency_steers_the_choice_when_speed_matters() -> None:
    world = build(
        tiers={
            "thrifty": (Tier.SLM, 0.05, 0.08, None),
            "quick": (Tier.SLM, 0.10, 0.16, None),
        },
        policy=RoutingPolicy.from_overrides({"cost_weight": 0.0, "latency_weight": 1.0}),
    )
    with world.client as client:
        latency: LatencyTracker = client.app.state.router.latency  # type: ignore[attr-defined]
        registry: ModelRegistry = client.app.state.registry  # type: ignore[attr-defined]
        latency.observe(registry.get("thrifty")[0].id, 8.0)
        latency.observe(registry.get("quick")[0].id, 0.3)

        response = ask(client, None, EASY)

    assert answer(response) == "answer from quick"


def test_cheaper_wins_within_a_tier_by_default() -> None:
    world = build(
        tiers={
            "dear": (Tier.SLM, 1.0, 2.0, None),
            "thrifty": (Tier.SLM, 0.05, 0.08, None),
        }
    )
    with world.client as client:
        response = ask(client, None, EASY)

    assert answer(response) == "answer from thrifty"


# ---------- quotas see the routed model ----------


def test_the_budget_is_charged_at_the_routed_models_price() -> None:
    world = build()
    with world.client as client:
        # Room for a worst-case cheap request (about $0.0000044) but not a frontier one ($0.00077).
        secret = client.issue_key(monthly_budget_usd=0.00001)

        easy = ask(client, secret, EASY, max_tokens=50)
        hard = ask(client, secret, HARD, max_tokens=50)

    assert easy.status_code == 200
    assert hard.status_code == 429
    assert hard.json()["error"]["code"] == "budget_exceeded"
    assert world.providers["top"].calls == 0  # refused before reaching the frontier model


def test_rate_limits_apply_to_auto_like_any_other_model() -> None:
    world = build()
    with world.client as client:
        secret = client.issue_key(rpm_limit=2)

        statuses = [ask(client, secret, EASY).status_code for _ in range(4)]

    assert statuses == [200, 200, 429, 429]


def test_token_limits_apply_to_auto() -> None:
    world = build()
    with world.client as client:
        secret = client.issue_key(tpm_limit=300)

        response = ask(client, secret, EASY)  # reserves ~320 tokens: too big for this key

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "request_too_large"


def test_retries_and_fallbacks_still_apply_to_the_chosen_model() -> None:
    world = build(fallbacks={"cheap": ["middle"]})
    world.providers["cheap"].fail_first = 99
    with world.client as client:
        response = ask(client, None, EASY)

    assert response.status_code == 200
    assert answer(response) == "answer from middle"  # cheap failed, so its fallback answered
    assert world.providers["cheap"].calls == 1


# ---------- who may use auto ----------


def test_a_key_must_be_allowed_auto_to_use_it() -> None:
    world = build()
    with world.client as client:
        secret = client.issue_key(allowed_models=["cheap"])

        response = ask(client, secret, EASY)

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "model_not_allowed"
    assert response.json()["error"]["message"].endswith("the model 'auto'.")
    assert world.served_by() == []


def test_a_key_allowed_only_auto_gets_the_routers_choice_but_not_a_free_pick() -> None:
    world = build()
    with world.client as client:
        secret = client.issue_key(allowed_models=["auto"])

        routed = ask(client, secret, HARD)
        direct = ask(client, secret, HARD, model="top")

    assert answer(routed) == "answer from top"  # auto may use any routable model
    assert direct.status_code == 403  # but the key cannot name one itself


def test_a_key_may_hold_auto_alongside_specific_models() -> None:
    world = build()
    with world.client as client:
        secret = client.issue_key(allowed_models=["cheap", "auto"])

        by_name = ask(client, secret, EASY, model="cheap")
        by_auto = ask(client, secret, HARD)
        other = ask(client, secret, EASY, model="middle")

    assert by_name.status_code == 200
    assert answer(by_auto) == "answer from top"
    assert other.status_code == 403


def test_a_pattern_that_matches_auto_grants_it() -> None:
    world = build()
    with world.client as client:
        secret = client.issue_key(allowed_models=["au*"])

        assert ask(client, secret, EASY).status_code == 200


# ---------- the model list ----------


def test_auto_is_listed_first() -> None:
    world = build()
    with world.client as client:
        ids = [m["id"] for m in client.get("/v1/models").json()["data"]]

    assert ids == ["auto", "cheap", "middle", "top"]


def test_auto_is_listed_only_to_keys_allowed_to_use_it() -> None:
    world = build()
    with world.client as client:
        limited = client.issue_key(allowed_models=["cheap"])
        only_auto = client.issue_key(allowed_models=["auto"])

        seen = {
            "limited": [
                m["id"]
                for m in client.get("/v1/models", headers=client.as_key(limited)).json()["data"]
            ],
            "only_auto": [
                m["id"]
                for m in client.get("/v1/models", headers=client.as_key(only_auto)).json()["data"]
            ],
        }

    assert seen == {"limited": ["cheap"], "only_auto": ["auto"]}


def test_without_any_tiered_models_auto_does_not_exist() -> None:
    world = build(tiers={"plain": (None, 0.0, 0.0, None)})
    with world.client as client:
        ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
        response = ask(client, None, EASY)
        direct = ask(client, None, EASY, model="plain")

    assert ids == ["plain"]
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"
    assert direct.status_code == 200  # ordinary models are unaffected


def test_the_name_is_case_sensitive() -> None:
    world = build()
    with world.client as client:
        response = ask(client, None, EASY, model="AUTO")

    assert response.status_code == 404


# ---------- context windows ----------


def test_a_long_but_easy_request_goes_to_the_model_that_can_hold_it() -> None:
    world = build(
        tiers={
            "tiny": (Tier.SLM, 0.0, 0.0, 400),
            "roomy": (Tier.FRONTIER, 3.0, 15.0, 200_000),
        }
    )
    with world.client as client:
        response = ask(client, None, "What is this? " + "filler " * 600)

    assert answer(response) == "answer from roomy"


def test_a_request_too_big_for_every_routable_model_is_a_context_length_error() -> None:
    world = build(tiers={"tiny": (Tier.SLM, 0.0, 0.0, 200), "small": (Tier.MID, 1.0, 1.0, 300)})
    with world.client as client:
        response = ask(client, None, "filler " * 1_000)

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert "maximum context length is 300 tokens" in error["message"]


# ---------- resolving, on its own ----------


def registry_of(*names: tuple[str, Tier | None, int | None]) -> ModelRegistry:
    return ModelRegistry(
        Deployment.create(
            model_config(name=name, model=f"{name}-up", tier=tier, context_window=window),
            FakeProvider(name),
        )
        for name, tier, window in names
    )


def raw(model: str, content: str = EASY) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {"model": model, "messages": [{"role": "user", "content": content}]}
    )


def test_resolving_an_ordinary_model_changes_nothing() -> None:
    registry, request = registry_of(("a", Tier.SLM, None)), raw("a")

    resolved, choice = resolve_model(request, registry, LatencyTracker())

    assert resolved is request and choice is None


def test_resolving_auto_names_the_chosen_model_without_touching_the_original() -> None:
    registry, request = registry_of(("a", Tier.SLM, None), ("b", Tier.FRONTIER, None)), raw("auto")

    resolved, choice = resolve_model(request, registry, LatencyTracker())

    assert resolved.model == "a" and choice is not None and choice.alias == "a"
    assert request.model == "auto"  # the caller's request is unchanged
    assert resolved.messages == request.messages


def test_resolving_with_no_tiers_looks_like_an_unknown_model() -> None:
    with pytest.raises(UnknownModelError) as excinfo:
        resolve_model(raw("auto"), registry_of(("a", None, None)), LatencyTracker())

    assert excinfo.value.alias == "auto"


def test_resolving_a_request_that_fits_nowhere_reports_the_largest_window() -> None:
    registry = registry_of(("a", Tier.SLM, 100), ("b", Tier.MID, 250))

    with pytest.raises(ContextLengthExceededError) as excinfo:
        resolve_model(raw("auto", "word " * 2_000), registry, LatencyTracker())

    assert excinfo.value.window == 250
    assert excinfo.value.required > 250


# ---------- the reserved name ----------


def test_a_configured_model_cannot_be_called_auto() -> None:
    body: dict[str, Any] = {
        "models": [
            {
                "name": "auto",
                "provider": "openai_compatible",
                "model": "m",
                "base_url": "https://api.example.com/v1",
            }
        ]
    }

    with pytest.raises(ConfigError, match="'auto' is reserved for smart routing"):
        parse_config(body, {})


def test_a_registry_cannot_be_built_with_a_model_called_auto() -> None:
    with pytest.raises(ConfigError, match="'auto' is reserved"):
        registry_of(("auto", Tier.SLM, None))


def test_a_config_file_naming_a_model_auto_prevents_startup(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "models:\n"
        "  - {name: auto, provider: openai_compatible, model: m, base_url: 'https://a.example/v1'}\n"
    )
    app = create_app(Settings(_env_file=None, config_path=config))

    with pytest.raises(ConfigError, match="reserved for smart routing"), AuthedClient(app):
        pass
