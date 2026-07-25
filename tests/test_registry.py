import dataclasses
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from app.core.model_config import ConfigError, ModelConfig, parse_models
from app.providers.fake import FakeProvider
from app.providers.openai_compat import OpenAICompatibleProvider
from app.router.deployment import Deployment, deployment_id
from app.router.registry import ModelRegistry, UnknownModelError, build_registry


def config(**overrides: Any) -> ModelConfig:
    fields: dict[str, Any] = {
        "name": "fast",
        "provider": "openai_compatible",
        "model": "llama-3.1-8b-instant",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": SecretStr("gsk-secret"),
        "weight": 1,
        "rpm": None,
        "tpm": None,
        "api_key_env": "GROQ_API_KEY",
    }
    fields.update(overrides)
    return ModelConfig(**fields)


def deployment(**overrides: Any) -> Deployment:
    return Deployment.create(config(**overrides), FakeProvider())


# ---------- stable ID ----------


def test_same_config_gives_same_id() -> None:
    assert deployment_id(config()) == deployment_id(config())
    assert deployment().id == deployment().id


def test_id_has_a_readable_prefix_and_fixed_length() -> None:
    value = deployment_id(config())

    assert value.startswith("dep_")
    assert len(value) == len("dep_") + 12


@pytest.mark.parametrize(
    "change",
    [
        {"name": "other"},
        {"model": "other-model"},
        {"provider": "anthropic"},
        {"base_url": "https://api.openai.com/v1"},
        {"api_key_env": "SECOND_GROQ_KEY"},
    ],
)
def test_identity_fields_change_the_id(change: dict[str, Any]) -> None:
    assert deployment_id(config(**change)) != deployment_id(config())


@pytest.mark.parametrize("change", [{"weight": 9}, {"rpm": 5}, {"tpm": 100}])
def test_tunables_do_not_change_the_id(change: dict[str, Any]) -> None:
    assert deployment_id(config(**change)) == deployment_id(config())


def test_secret_value_is_not_part_of_the_id() -> None:
    first = deployment_id(config(api_key=SecretStr("one")))
    second = deployment_id(config(api_key=SecretStr("two")))

    assert first == second
    assert "gsk-secret" not in first


def test_id_comes_from_yaml_via_parse_models() -> None:
    raw = {
        "models": [
            {
                "name": "fast",
                "provider": "openai_compatible",
                "model": "m",
                "base_url": "https://api.example.com/v1",
                "api_key": "os.environ/KEY_A",
            }
        ]
    }

    first = parse_models(raw, {"KEY_A": "v1"})[0]
    second = parse_models(raw, {"KEY_A": "v2"})[0]

    assert first.api_key_env == "KEY_A"
    assert deployment_id(first) == deployment_id(second)


def test_deployment_exposes_convenient_accessors() -> None:
    item = deployment(weight=4)

    assert item.alias == "fast"
    assert item.upstream_model == "llama-3.1-8b-instant"
    assert item.weight == 4
    with pytest.raises(dataclasses.FrozenInstanceError):
        item.id = "x"  # type: ignore[misc]


# ---------- registry ----------


def test_groups_deployments_by_alias_in_config_order() -> None:
    registry = ModelRegistry(
        [
            deployment(name="fast", model="a"),
            deployment(name="smart", model="b", provider="anthropic", base_url=None),
            deployment(name="fast", model="c"),
        ]
    )

    assert registry.aliases() == ["fast", "smart"]
    assert [d.upstream_model for d in registry.get("fast")] == ["a", "c"]
    assert [d.upstream_model for d in registry.get("smart")] == ["b"]
    assert len(registry) == 3
    assert "fast" in registry
    assert "nope" not in registry


def test_unknown_alias_raises_unknown_model() -> None:
    registry = ModelRegistry([deployment()])

    with pytest.raises(UnknownModelError) as excinfo:
        registry.get("gpt-9")

    assert excinfo.value.alias == "gpt-9"
    assert "gpt-9" in str(excinfo.value)


def test_identical_entries_are_rejected() -> None:
    with pytest.raises(ConfigError, match="duplicate deployment for model 'fast'"):
        ModelRegistry([deployment(), deployment(weight=5)])


def test_same_model_with_different_keys_is_allowed() -> None:
    registry = ModelRegistry([deployment(), deployment(api_key_env="SECOND_GROQ_KEY")])

    assert len(registry.get("fast")) == 2
    assert len({d.id for d in registry.get("fast")}) == 2


def test_empty_registry_is_valid_but_empty() -> None:
    registry = ModelRegistry([])

    assert registry.aliases() == []
    assert len(registry) == 0


def test_lookup_cannot_mutate_the_registry() -> None:
    registry = ModelRegistry([deployment()])

    assert isinstance(registry.get("fast"), tuple)


def test_build_registry_creates_real_providers() -> None:
    registry = build_registry([config()], httpx.AsyncClient())

    item = registry.get("fast")[0]
    assert isinstance(item.provider, OpenAICompatibleProvider)
    assert item.id == deployment_id(config())
