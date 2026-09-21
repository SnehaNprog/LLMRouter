from pathlib import Path
from typing import Any

import pytest

from app.core.model_config import ConfigError, load_config, parse_config, tier_problems
from app.core.tiers import TIERS, Tier
from app.providers.fake import FakeProvider
from app.router.deployment import Deployment, deployment_id
from app.router.registry import ModelRegistry, UnknownModelError
from tests.conftest import AuthedClient, model_config

# ---------- the tiers themselves ----------


def test_there_are_three_tiers_weakest_first() -> None:
    assert [t.value for t in TIERS] == ["slm", "mid", "frontier"]
    assert list(TIERS) == [Tier.SLM, Tier.MID, Tier.FRONTIER]


def test_tiers_compare_by_strength_not_alphabetically() -> None:
    assert Tier.SLM < Tier.MID < Tier.FRONTIER
    assert Tier.FRONTIER > Tier.MID > Tier.SLM
    assert Tier.SLM <= Tier.SLM <= Tier.MID
    assert Tier.FRONTIER >= Tier.FRONTIER >= Tier.MID
    assert sorted([Tier.FRONTIER, Tier.SLM, Tier.MID]) == [Tier.SLM, Tier.MID, Tier.FRONTIER]
    assert max(Tier.SLM, Tier.FRONTIER, Tier.MID) is Tier.FRONTIER


def test_every_comparison_agrees_with_the_ranking_for_every_pair() -> None:
    for left in TIERS:
        for right in TIERS:
            assert (left < right) == (left.rank < right.rank)
            assert (left <= right) == (left.rank <= right.rank)
            assert (left > right) == (left.rank > right.rank)
            assert (left >= right) == (left.rank >= right.rank)


def test_alphabetical_order_would_get_it_backwards() -> None:
    assert "frontier" < "mid" < "slm"  # which is why Tier overrides the string comparisons


def test_tiers_are_still_plain_strings() -> None:
    assert isinstance(Tier.SLM, str)
    assert Tier.SLM.value == "slm"
    assert Tier("mid") is Tier.MID
    assert f"{Tier.FRONTIER}" == "frontier"


def test_unknown_tier_names_are_rejected() -> None:
    with pytest.raises(ValueError):
        Tier("gigantic")


def test_escalation_walks_up_and_stops_at_the_top() -> None:
    assert Tier.SLM.stronger() is Tier.MID
    assert Tier.MID.stronger() is Tier.FRONTIER
    assert Tier.FRONTIER.stronger() is None


def test_de_escalation_walks_down_and_stops_at_the_bottom() -> None:
    assert Tier.FRONTIER.weaker() is Tier.MID
    assert Tier.MID.weaker() is Tier.SLM
    assert Tier.SLM.weaker() is None


@pytest.mark.parametrize(
    "operate",
    [
        lambda t: t < 3,
        lambda t: t <= 3,
        lambda t: t > 3,
        lambda t: t >= 3,
    ],
)
def test_comparing_with_a_non_tier_is_not_supported(operate: Any) -> None:
    with pytest.raises(TypeError):
        operate(Tier.SLM)


# ---------- in the config file ----------


def entry(name: str, tier: str | None = None, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": name,
        "provider": "openai_compatible",
        "model": f"{name}-model",
        "base_url": "https://api.example.com/v1",
        **extra,
    }
    if tier is not None:
        body["tier"] = tier
    return body


def models_of(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"models": list(entries)}


def test_a_tier_is_read_from_the_config() -> None:
    config = parse_config(
        models_of(entry("a", "slm"), entry("b", "mid"), entry("c", "frontier")), {}
    )

    assert [m.tier for m in config.models] == [Tier.SLM, Tier.MID, Tier.FRONTIER]


def test_the_tier_is_optional() -> None:
    assert parse_config(models_of(entry("a")), {}).models[0].tier is None


@pytest.mark.parametrize("bad", ["gigantic", "SLM", "", 3])
def test_unknown_tiers_are_rejected_with_their_position(bad: Any) -> None:
    with pytest.raises(ConfigError, match=r"models\[1\]\.tier"):
        parse_config(models_of(entry("a"), entry("b", bad)), {})


def test_entries_of_one_model_may_repeat_the_same_tier() -> None:
    config = parse_config(models_of(entry("a", "mid"), entry("a", "mid", model="other")), {})

    assert [m.tier for m in config.models] == [Tier.MID, Tier.MID]


def test_entries_of_one_model_must_agree_on_the_tier() -> None:
    with pytest.raises(ConfigError) as excinfo:
        parse_config(models_of(entry("a", "slm"), entry("b", "mid"), entry("a", "frontier")), {})

    message = str(excinfo.value)
    assert "models[2] ('a')" in message
    assert "'frontier' conflicts with tier 'slm' of models[0]" in message


def test_declaring_a_tier_on_only_some_entries_of_a_model_is_a_conflict() -> None:
    with pytest.raises(ConfigError, match=r"tier \(none\) conflicts with tier 'mid'"):
        parse_config(models_of(entry("a", "mid"), entry("a", None, model="other")), {})


def test_tier_conflicts_are_reported_alongside_other_problems() -> None:
    bad = models_of(entry("a", "slm"), entry("a", "mid"))
    bad["fallbacks"] = {"a": ["ghost"]}

    with pytest.raises(ConfigError) as excinfo:
        parse_config(bad, {})

    assert "conflicts with tier" in str(excinfo.value)
    assert "unknown model 'ghost'" in str(excinfo.value)


def test_tier_problems_works_on_loaded_configs() -> None:
    clash = [model_config(name="a", tier=Tier.SLM), model_config(name="a", tier=Tier.MID)]

    assert len(tier_problems(clash)) == 1
    assert tier_problems([model_config(name="a"), model_config(name="b", tier=Tier.MID)]) == []


def test_a_tier_does_not_change_a_deployments_identity() -> None:
    assert deployment_id(model_config(tier=Tier.SLM)) == deployment_id(model_config(tier=Tier.MID))


def test_the_shipped_example_uses_all_three_tiers() -> None:
    config = load_config(
        Path(__file__).parent.parent / "config.example.yaml",
        {"GROQ_API_KEY": "a", "ANTHROPIC_API_KEY": "b"},
    )

    tiers = {m.name: m.tier for m in config.models}
    assert tiers == {
        "fast": Tier.SLM,
        "local": Tier.SLM,
        "balanced": Tier.MID,
        "smart": Tier.FRONTIER,
    }


def test_the_example_prices_rise_with_the_tier() -> None:
    config = load_config(
        Path(__file__).parent.parent / "config.example.yaml",
        {"GROQ_API_KEY": "a", "ANTHROPIC_API_KEY": "b"},
    )

    price = {m.name: m.output_cost_per_mtok for m in config.models}
    assert price["fast"] < price["balanced"] < price["smart"]


# ---------- in the registry ----------


def registry(*names: tuple[str, Tier | None]) -> ModelRegistry:
    return ModelRegistry(
        Deployment.create(
            model_config(name=name, model=f"{name}-up", tier=tier), FakeProvider(name)
        )
        for name, tier in names
    )


def test_a_models_tier_can_be_looked_up() -> None:
    reg = registry(("fast", Tier.SLM), ("smart", Tier.FRONTIER), ("plain", None))

    assert reg.tier_of("fast") is Tier.SLM
    assert reg.tier_of("smart") is Tier.FRONTIER
    assert reg.tier_of("plain") is None


def test_looking_up_the_tier_of_an_unknown_model_fails_clearly() -> None:
    with pytest.raises(UnknownModelError):
        registry(("fast", Tier.SLM)).tier_of("nope")


def test_models_are_listed_by_tier_in_config_order() -> None:
    reg = registry(
        ("b", Tier.SLM), ("z", Tier.FRONTIER), ("a", Tier.SLM), ("m", Tier.MID), ("x", None)
    )

    assert reg.aliases_in_tier(Tier.SLM) == ["b", "a"]
    assert reg.aliases_in_tier(Tier.MID) == ["m"]
    assert reg.aliases_in_tier(Tier.FRONTIER) == ["z"]


def test_an_empty_tier_lists_nothing() -> None:
    assert registry(("a", Tier.SLM)).aliases_in_tier(Tier.FRONTIER) == []


def test_the_tier_overview_is_weakest_first_and_skips_empty_and_untiered() -> None:
    reg = registry(("z", Tier.FRONTIER), ("a", Tier.SLM), ("x", None))

    overview = reg.tiers()

    assert list(overview) == [Tier.SLM, Tier.FRONTIER]
    assert overview == {Tier.SLM: ["a"], Tier.FRONTIER: ["z"]}


def test_a_registry_without_tiers_has_an_empty_overview() -> None:
    assert registry(("a", None), ("b", None)).tiers() == {}


def test_several_deployments_of_one_model_count_once() -> None:
    reg = ModelRegistry(
        [
            Deployment.create(model_config(name="a", model="one", tier=Tier.MID), FakeProvider()),
            Deployment.create(model_config(name="a", model="two", tier=Tier.MID), FakeProvider()),
        ]
    )

    assert reg.aliases_in_tier(Tier.MID) == ["a"]


def test_a_registry_rejects_models_with_conflicting_tiers() -> None:
    with pytest.raises(ConfigError, match="inconsistent tiers"):
        ModelRegistry(
            [
                Deployment.create(
                    model_config(name="a", model="one", tier=Tier.SLM), FakeProvider()
                ),
                Deployment.create(
                    model_config(name="a", model="two", tier=Tier.MID), FakeProvider()
                ),
            ]
        )


# ---------- loaded at startup ----------


def test_the_running_gateway_knows_the_tiers_from_the_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.core.config import Settings
    from app.main import create_app

    config = tmp_path / "config.yaml"
    config.write_text(
        "models:\n"
        "  - {name: fast, tier: slm, provider: openai_compatible, model: m1, base_url: 'https://a.example/v1'}\n"
        "  - {name: smart, tier: frontier, provider: openai_compatible, model: m2, base_url: 'https://a.example/v1'}\n"
        "  - {name: other, provider: openai_compatible, model: m3, base_url: 'https://a.example/v1'}\n"
    )
    app = create_app(Settings(_env_file=None, config_path=config))

    with AuthedClient(app):
        overview = app.state.registry.tiers()

    assert overview == {Tier.SLM: ["fast"], Tier.FRONTIER: ["smart"]}


def test_a_config_with_conflicting_tiers_prevents_startup(tmp_path: Path) -> None:
    from app.core.config import Settings
    from app.main import create_app

    config = tmp_path / "config.yaml"
    config.write_text(
        "models:\n"
        "  - {name: a, tier: slm, provider: openai_compatible, model: m1, base_url: 'https://a.example/v1'}\n"
        "  - {name: a, tier: mid, provider: openai_compatible, model: m2, base_url: 'https://a.example/v1'}\n"
    )
    app = create_app(Settings(_env_file=None, config_path=config))

    with pytest.raises(ConfigError, match="conflicts with tier"), AuthedClient(app):
        pass
