import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from app.core.config import Settings
from app.core.model_config import ConfigError, parse_config
from app.core.tiers import Tier
from app.main import create_app
from app.providers.fake import FakeProvider
from app.router.deployment import Deployment
from app.router.latency import LatencyTracker
from app.router.registry import ModelRegistry
from app.routing.policy import DEFAULT_POLICY, RoutingPolicy
from app.routing.score import ComplexityScore, score_request
from app.routing.selector import NoRoutableModelError, choose_model
from app.schemas.chat import ChatCompletionRequest
from tests.conftest import AuthedClient, model_config


def req(content: str = "Hi", **extra: Any) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {"model": "auto", "messages": [{"role": "user", "content": content}], **extra}
    )


def pinned(value: float) -> ComplexityScore:
    """A complexity score fixed by hand, so tier choices can be tested in isolation."""
    return ComplexityScore(value=value, contributions={}, small_talk=False)


def dep(
    name: str,
    tier: Tier | None,
    input_price: float = 0.0,
    output_price: float = 0.0,
    window: int | None = None,
    weight: int = 1,
    model: str | None = None,
) -> Deployment:
    return Deployment.create(
        model_config(
            name=name,
            model=model or f"{name}-up",
            tier=tier,
            input_cost_per_mtok=input_price,
            output_cost_per_mtok=output_price,
            context_window=window,
            weight=weight,
        ),
        FakeProvider(name),
    )


def ladder() -> ModelRegistry:
    return ModelRegistry(
        [
            dep("cheap", Tier.SLM, 0.05, 0.08),
            dep("middle", Tier.MID, 0.6, 0.8),
            dep("top", Tier.FRONTIER, 3.0, 15.0),
        ]
    )


def choose(registry: ModelRegistry, complexity: float, **kwargs: Any) -> Any:
    request = kwargs.pop("request", req())
    return choose_model(
        request,
        registry,
        kwargs.pop("latency", LatencyTracker()),
        score=pinned(complexity),
        **kwargs,
    )


# ---------- the policy ----------


@pytest.mark.parametrize(
    ("complexity", "tier"),
    [
        (0.0, Tier.SLM),
        (0.35, Tier.SLM),  # a ceiling belongs to its own tier
        (0.3501, Tier.MID),
        (0.65, Tier.MID),
        (0.6501, Tier.FRONTIER),
        (1.0, Tier.FRONTIER),
    ],
)
def test_the_complexity_ceilings_decide_the_weakest_tier_that_will_do(
    complexity: float, tier: Tier
) -> None:
    assert DEFAULT_POLICY.required_tier(complexity) is tier


def test_the_ceilings_can_be_moved() -> None:
    cautious = RoutingPolicy.from_overrides({"slm_ceiling": 0.1, "mid_ceiling": 0.3})

    assert cautious.required_tier(0.2) is Tier.MID
    assert cautious.required_tier(0.31) is Tier.FRONTIER
    assert DEFAULT_POLICY.required_tier(0.2) is Tier.SLM


def test_the_assumed_answer_length_follows_max_tokens_within_reason() -> None:
    assert DEFAULT_POLICY.completion_estimate(req()) == 300
    assert DEFAULT_POLICY.completion_estimate(req(max_tokens=50)) == 50
    assert DEFAULT_POLICY.completion_estimate(req(max_tokens=100_000)) == 4_000  # capped
    assert DEFAULT_POLICY.completion_estimate(req(max_tokens=100, n=3)) == 300


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"slm_ceiling": 0.7, "mid_ceiling": 0.5}, "must be below mid_ceiling"),
        ({"slm_ceiling": 0.5, "mid_ceiling": 0.5}, "must be below mid_ceiling"),
        ({"slm_ceiling": 0}, "slm_ceiling: must be above 0"),
        ({"mid_ceiling": 1.5}, "mid_ceiling: must be above 0 and at most 1"),
        ({"cost_weight": -1}, "cost_weight: must not be negative"),
        ({"cost_weight": 0, "latency_weight": 0}, "cannot both be zero"),
        ({"expected_completion_tokens": 0}, "expected_completion_tokens: must be at least 1"),
    ],
)
def test_nonsense_policies_are_rejected_clearly(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RoutingPolicy.from_overrides(overrides)


def test_a_partial_override_keeps_the_other_defaults() -> None:
    policy = RoutingPolicy.from_overrides({"cost_weight": 1.0})

    assert policy.cost_weight == 1.0
    assert (policy.slm_ceiling, policy.latency_weight) == (0.35, 0.3)


# ---------- choosing a tier ----------


def test_an_easy_request_goes_to_the_small_model() -> None:
    assert choose(ladder(), 0.1).alias == "cheap"


def test_a_middling_request_goes_to_the_mid_tier() -> None:
    assert choose(ladder(), 0.5).alias == "middle"


def test_a_hard_request_goes_to_the_frontier_model() -> None:
    assert choose(ladder(), 0.9).alias == "top"


def test_stronger_models_are_candidates_too_but_the_cheaper_one_wins() -> None:
    choice = choose(ladder(), 0.5)

    assert [c.alias for c in choice.candidates] == ["middle", "top"]
    assert choice.clears_bar and choice.required_tier is Tier.MID


def test_a_missing_tier_is_skipped_for_the_next_one_up() -> None:
    registry = ModelRegistry([dep("cheap", Tier.SLM), dep("top", Tier.FRONTIER, 3, 15)])

    assert choose(registry, 0.5).alias == "top"  # no mid-tier model exists


def test_when_nothing_is_strong_enough_the_strongest_available_is_used_and_flagged() -> None:
    registry = ModelRegistry([dep("cheap", Tier.SLM), dep("also-cheap", Tier.SLM)])

    choice = choose(registry, 0.95)

    assert not choice.clears_bar
    assert choice.required_tier is Tier.FRONTIER
    assert choice.tier is Tier.SLM
    assert any("no model of tier frontier" in line for line in choice.reasons)


def test_a_request_never_goes_to_a_weaker_tier_than_it_needs_when_a_strong_one_exists() -> None:
    for complexity in (0.4, 0.5, 0.64):
        assert choose(ladder(), complexity).tier >= Tier.MID


def test_models_without_a_tier_are_never_chosen() -> None:
    registry = ModelRegistry([dep("plain", None, 0, 0), dep("top", Tier.FRONTIER, 3, 15)])

    assert choose(registry, 0.0).alias == "top"
    assert {c.alias for c in choose(registry, 0.0).candidates} == {"top"}


def test_a_registry_with_no_tiers_cannot_be_routed() -> None:
    registry = ModelRegistry([dep("plain", None)])

    with pytest.raises(NoRoutableModelError, match="no model has a tier"):
        choose(registry, 0.5)


# ---------- cost ----------


def test_within_a_tier_the_cheaper_model_wins() -> None:
    registry = ModelRegistry([dep("dear", Tier.SLM, 1.0, 2.0), dep("thrifty", Tier.SLM, 0.1, 0.2)])

    choice = choose(registry, 0.1)

    assert [c.alias for c in choice.candidates] == ["thrifty", "dear"]
    assert choice.candidates[0].expected_cost_usd < choice.candidates[1].expected_cost_usd


def test_the_expected_cost_is_prompt_plus_assumed_answer_at_the_models_prices() -> None:
    registry = ModelRegistry([dep("m", Tier.SLM, 1.0, 2.0)])

    candidate = choose(registry, 0.1).candidates[0]

    # "Hi" is 8 prompt tokens; the default assumed answer is 300 tokens.
    assert candidate.expected_cost_usd == pytest.approx((8 * 1.0 + 300 * 2.0) / 1_000_000)


def test_a_longer_requested_answer_raises_the_expected_cost() -> None:
    registry = ModelRegistry([dep("m", Tier.SLM, 1.0, 2.0)])

    short = choose(registry, 0.1, request=req(max_tokens=50)).candidates[0].expected_cost_usd
    long = choose(registry, 0.1, request=req(max_tokens=2000)).candidates[0].expected_cost_usd

    assert long > short * 10


def test_a_free_model_beats_a_priced_one_of_the_same_tier() -> None:
    registry = ModelRegistry([dep("paid", Tier.SLM, 0.05, 0.08), dep("local", Tier.SLM)])

    assert choose(registry, 0.1).alias == "local"


def test_several_deployments_of_one_model_are_priced_by_their_weights() -> None:
    registry = ModelRegistry(
        [
            dep("blend", Tier.SLM, 1.0, 0.0, model="a", weight=3),
            dep("blend", Tier.SLM, 5.0, 0.0, model="b", weight=1),
            dep("steady", Tier.SLM, 2.0, 0.0),
        ]
    )

    costs = {c.alias: c.expected_cost_usd for c in choose(registry, 0.1).candidates}

    assert costs["blend"] == pytest.approx(8 * 2.0 / 1e6)  # (3*1 + 1*5) / 4 = 2 per million
    assert costs["blend"] == pytest.approx(costs["steady"])


def test_ties_fall_back_to_config_order() -> None:
    registry = ModelRegistry([dep("first", Tier.SLM, 1, 1), dep("second", Tier.SLM, 1, 1)])

    assert [c.alias for c in choose(registry, 0.1).candidates] == ["first", "second"]


# ---------- latency ----------


def timed(registry: ModelRegistry, **seconds: float) -> LatencyTracker:
    tracker = LatencyTracker()
    for alias, value in seconds.items():
        for d in registry.get(alias):
            tracker.observe(d.id, value)
    return tracker


def test_measured_latency_can_outweigh_a_price_advantage() -> None:
    registry = ModelRegistry(
        [dep("slow-cheap", Tier.SLM, 0.1, 0.1), dep("fast-dear", Tier.SLM, 0.2, 0.2)]
    )
    tracker = timed(registry, **{"slow-cheap": 9.0, "fast-dear": 0.2})
    speed_matters = RoutingPolicy.from_overrides({"cost_weight": 0.1, "latency_weight": 0.9})

    assert choose(registry, 0.1, latency=tracker, policy=speed_matters).alias == "fast-dear"
    assert choose(registry, 0.1, latency=tracker).alias == "slow-cheap"  # the default favours price


def test_with_latency_ignored_the_cheapest_always_wins() -> None:
    registry = ModelRegistry(
        [dep("slow-cheap", Tier.SLM, 0.1, 0.1), dep("fast-dear", Tier.SLM, 0.2, 0.2)]
    )
    tracker = timed(registry, **{"slow-cheap": 9.0, "fast-dear": 0.2})
    cost_only = RoutingPolicy.from_overrides({"cost_weight": 1.0, "latency_weight": 0.0})

    assert choose(registry, 0.1, latency=tracker, policy=cost_only).alias == "slow-cheap"


def test_with_cost_ignored_the_fastest_always_wins() -> None:
    registry = ModelRegistry(
        [dep("slow-cheap", Tier.SLM, 0.1, 0.1), dep("fast-dear", Tier.SLM, 9, 9)]
    )
    tracker = timed(registry, **{"slow-cheap": 9.0, "fast-dear": 0.2})
    speed_only = RoutingPolicy.from_overrides({"cost_weight": 0.0, "latency_weight": 1.0})

    assert choose(registry, 0.1, latency=tracker, policy=speed_only).alias == "fast-dear"


def test_unmeasured_models_use_their_tiers_default_and_say_so() -> None:
    registry = ModelRegistry([dep("a", Tier.SLM), dep("b", Tier.FRONTIER)])
    tracker = timed(registry, b=2.0)

    candidates = {c.alias: c for c in choose(registry, 0.0, latency=tracker).candidates}

    assert candidates["a"].expected_latency_s == pytest.approx(0.6)  # the SLM default
    assert not candidates["a"].latency_measured
    assert candidates["b"].expected_latency_s == pytest.approx(2.0)
    assert candidates["b"].latency_measured


def test_a_partly_measured_model_blends_data_and_default_by_weight() -> None:
    registry = ModelRegistry(
        [dep("m", Tier.SLM, model="x", weight=1), dep("m", Tier.SLM, model="y", weight=1)]
    )
    tracker = LatencyTracker()
    tracker.observe(registry.get("m")[0].id, 2.0)  # the other deployment has no data yet

    candidate = choose(registry, 0.0, latency=tracker).candidates[0]

    assert candidate.expected_latency_s == pytest.approx((2.0 + 0.6) / 2)
    assert not candidate.latency_measured


def test_a_slow_local_model_can_lose_to_a_quick_hosted_one() -> None:
    registry = ModelRegistry([dep("local", Tier.SLM), dep("hosted", Tier.SLM, 0.05, 0.08)])
    tracker = timed(registry, local=12.0, hosted=0.4)
    balanced = RoutingPolicy.from_overrides({"cost_weight": 0.3, "latency_weight": 0.7})

    assert choose(registry, 0.1, latency=tracker, policy=balanced).alias == "hosted"


def test_a_lone_candidate_has_nothing_to_be_compared_with() -> None:
    choice = choose(ModelRegistry([dep("only", Tier.SLM, 1, 1)]), 0.1)

    assert choice.alias == "only"
    assert len(choice.candidates) == 1
    assert choice.candidates[0].utility == pytest.approx(
        1.0
    )  # utilities are relative: it is the max


# ---------- limits on what may be chosen ----------


def test_a_model_whose_window_cannot_hold_the_request_is_skipped() -> None:
    registry = ModelRegistry(
        [dep("tiny", Tier.SLM, window=100), dep("roomy", Tier.SLM, 1, 1, window=100_000)]
    )
    long_ask = req("word " * 400)  # about 500 tokens

    choice = choose(registry, 0.1, request=long_ask)

    assert [c.alias for c in choice.candidates] == ["roomy"]


def test_the_assumed_answer_counts_towards_the_window() -> None:
    registry = ModelRegistry(
        [dep("snug", Tier.SLM, window=300), dep("roomy", Tier.SLM, 1, 1, window=10_000)]
    )

    default_answer = choose(registry, 0.1)  # 8 prompt + 300 assumed answer > 300
    short_answer = choose(registry, 0.1, request=req(max_tokens=20))  # 8 + 20 fits easily

    assert [c.alias for c in default_answer.candidates] == ["roomy"]
    assert {c.alias for c in short_answer.candidates} == {"snug", "roomy"}


def test_a_model_is_usable_if_any_of_its_deployments_fits() -> None:
    registry = ModelRegistry(
        [
            dep("mixed", Tier.SLM, window=100, model="small"),
            dep("mixed", Tier.SLM, window=100_000, model="big"),
        ]
    )

    assert choose(registry, 0.1, request=req("word " * 400)).alias == "mixed"


def test_nothing_fitting_is_reported_as_such() -> None:
    registry = ModelRegistry(
        [dep("tiny", Tier.SLM, window=50), dep("small", Tier.FRONTIER, window=60)]
    )

    with pytest.raises(NoRoutableModelError, match="fits no routable model's context window"):
        choose(registry, 0.1)


def test_a_key_is_only_routed_to_models_it_may_use() -> None:
    allow = lambda alias: alias != "middle"  # noqa: E731

    choice = choose(ladder(), 0.5, allow=allow)

    assert [c.alias for c in choice.candidates] == ["top"]


def test_when_the_allowed_models_are_too_weak_the_best_allowed_is_used_and_flagged() -> None:
    choice = choose(ladder(), 0.9, allow=lambda alias: alias == "cheap")

    assert choice.alias == "cheap" and not choice.clears_bar


def test_a_key_allowed_nothing_routable_gets_a_clear_error() -> None:
    with pytest.raises(NoRoutableModelError, match="may not use any of the routable models"):
        choose(ladder(), 0.5, allow=lambda alias: False)


# ---------- explaining a choice ----------


def test_the_choice_carries_the_complexity_and_a_readable_account() -> None:
    request = req("Compare A and B, analyze the trade-offs step by step, and justify.")
    registry = ladder()

    choice = choose_model(request, registry, LatencyTracker())

    assert choice.complexity == score_request(request)
    assert "reasoning" in " ".join(choice.reasons)
    assert choice.alias in " ".join(choice.reasons)
    assert choice.required_tier.value in choice.reasons[0]


def test_the_ranking_is_best_first_and_the_choice_is_its_head() -> None:
    choice = choose(ladder(), 0.0)

    utilities = [c.utility for c in choice.candidates]
    assert utilities == sorted(utilities)
    assert choice.alias == choice.candidates[0].alias
    assert choice.tier is choice.candidates[0].tier


def test_a_real_prompt_uses_the_registrys_own_scoring_and_policy() -> None:
    registry = ModelRegistry(
        [dep("cheap", Tier.SLM, 0.05, 0.08), dep("top", Tier.FRONTIER, 3, 15)],
        routing_policy=RoutingPolicy.from_overrides({"slm_ceiling": 0.01, "mid_ceiling": 0.02}),
    )

    # Under these ceilings even a plain factual question needs the frontier tier.
    assert (
        choose_model(req("What is the capital of France?"), registry, LatencyTracker()).alias
        == "top"
    )


def test_selection_is_deterministic_and_changes_nothing() -> None:
    registry, tracker, request = ladder(), LatencyTracker(), req("Compare the options.")
    before = (registry.aliases(), tracker.snapshot(), request.model_dump())

    first = choose_model(request, registry, tracker)
    second = choose_model(request, registry, tracker)

    assert first == second
    assert (registry.aliases(), tracker.snapshot(), request.model_dump()) == before


# ---------- invariants over many random setups ----------


def allowing_all_but(banned: set[str]) -> Callable[[str], bool]:
    def allow(alias: str) -> bool:
        return alias not in banned

    return allow


def test_the_choice_is_always_consistent_across_random_registries() -> None:
    rng = random.Random(2026)
    for round_number in range(300):
        deployments = []
        for index in range(rng.randint(1, 7)):
            deployments.append(
                dep(
                    f"m{index}",
                    rng.choice([Tier.SLM, Tier.MID, Tier.FRONTIER, None]),
                    rng.choice([0.0, 0.05, 0.5, 3.0, 10.0]),
                    rng.choice([0.0, 0.08, 1.0, 15.0]),
                    rng.choice([None, 200, 5_000, 200_000]),
                    rng.randint(1, 4),
                )
            )
        registry = ModelRegistry(deployments)
        tracker = LatencyTracker()
        for d in deployments:
            if rng.random() < 0.6:
                tracker.observe(d.id, rng.uniform(0.1, 10.0))
        banned = {d.alias for d in deployments if rng.random() < 0.2}
        complexity = rng.random()
        request = req("word " * rng.randint(1, 200), max_tokens=rng.choice([None, 20, 800]))

        allow = allowing_all_but(banned)
        try:
            choice = choose_model(request, registry, tracker, score=pinned(complexity), allow=allow)
        except NoRoutableModelError:
            continue

        needed = choice.required_tier
        utilities = [c.utility for c in choice.candidates]
        assert utilities == sorted(utilities), round_number
        assert choice.alias not in banned
        assert all(registry.tier_of(c.alias) is not None for c in choice.candidates)
        if choice.clears_bar:
            assert all(c.tier >= needed for c in choice.candidates), round_number
        else:
            strongest = max(c.tier for c in choice.candidates)
            assert all(c.tier == strongest for c in choice.candidates)
            assert strongest < needed
        assert choice == choose_model(
            request, registry, tracker, score=pinned(complexity), allow=allow
        )


# ---------- from the config file ----------


def models(**routing: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "models": [
            {
                "name": "a",
                "provider": "openai_compatible",
                "model": "m",
                "base_url": "https://api.example.com/v1",
            }
        ]
    }
    if routing:
        body["routing"] = routing
    return body


def test_the_routing_policy_defaults_when_not_configured() -> None:
    assert parse_config(models(), {}).policy == DEFAULT_POLICY


def test_the_routing_policy_is_read_from_the_config() -> None:
    config = parse_config(
        models(
            tier_ceilings={"slm": 0.2, "mid": 0.5},
            objective={"cost": 0.9, "latency": 0.1},
            expected_completion_tokens=500,
        ),
        {},
    )

    policy = config.policy
    assert (policy.slm_ceiling, policy.mid_ceiling) == (0.2, 0.5)
    assert (policy.cost_weight, policy.latency_weight) == (0.9, 0.1)
    assert policy.expected_completion_tokens == 500


def test_a_partly_configured_policy_keeps_the_other_defaults() -> None:
    policy = parse_config(models(tier_ceilings={"mid": 0.8}), {}).policy

    assert (policy.slm_ceiling, policy.mid_ceiling) == (0.35, 0.8)
    assert policy.cost_weight == 0.7


@pytest.mark.parametrize(
    ("routing", "message"),
    [
        ({"tier_ceilings": {"slm": 0.8, "mid": 0.4}}, "routing: slm_ceiling .* below mid_ceiling"),
        ({"tier_ceilings": {"slm": 0}}, "routing: slm_ceiling: must be above 0"),
        ({"tier_ceilings": {"frontier": 0.9}}, r"routing\.tier_ceilings"),
        ({"tier_ceilings": {"slm": "low"}}, r"routing\.tier_ceilings"),
        ({"objective": {"cost": 0, "latency": 0}}, "cannot both be zero"),
        ({"objective": {"cost": -1}}, "cost_weight: must not be negative"),
        ({"objective": {"speed": 1}}, r"routing\.objective"),
        ({"expected_completion_tokens": 0}, "expected_completion_tokens: must be at least 1"),
        ({"expected_completion_tokens": "lots"}, r"routing\.expected_completion_tokens"),
    ],
)
def test_bad_routing_policies_stop_startup_with_a_clear_message(
    routing: dict[str, Any], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config(models(**routing), {})


def test_the_running_gateway_carries_the_configured_policy(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "models:\n"
        "  - {name: a, provider: openai_compatible, model: m, base_url: 'https://a.example/v1'}\n"
        "routing:\n"
        "  tier_ceilings: {slm: 0.2, mid: 0.4}\n"
        "  objective: {cost: 1, latency: 0}\n"
    )
    app = create_app(Settings(_env_file=None, config_path=config))

    with AuthedClient(app):
        policy = app.state.registry.routing_policy

    assert (policy.slm_ceiling, policy.mid_ceiling, policy.latency_weight) == (0.2, 0.4, 0.0)
