from pathlib import Path

import pytest

from app.core.model_config import ConfigError, load_models, parse_models

ENV = {"GROQ_API_KEY": "gsk-secret-value", "ANTHROPIC_API_KEY": "sk-ant-secret"}

VALID = {
    "models": [
        {
            "name": "fast",
            "provider": "openai_compatible",
            "model": "llama-3.1-8b-instant",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": "os.environ/GROQ_API_KEY",
            "weight": 3,
            "rpm": 30,
        },
        {
            "name": "smart",
            "provider": "anthropic",
            "model": "claude-sonnet-5-5",
            "api_key": "os.environ/ANTHROPIC_API_KEY",
        },
    ]
}


def with_entry(**overrides: object) -> dict[str, object]:
    entry = {**VALID["models"][0], **overrides}  # type: ignore[dict-item]
    return {"models": [entry]}


def problems(raw: object, env: dict[str, str] | None = None) -> str:
    with pytest.raises(ConfigError) as excinfo:
        parse_models(raw, ENV if env is None else env)
    return str(excinfo.value)


def test_valid_config_is_parsed_with_defaults() -> None:
    models = parse_models(VALID, ENV)

    assert [m.name for m in models] == ["fast", "smart"]
    assert models[0].weight == 3
    assert models[0].rpm == 30
    assert models[1].weight == 1
    assert models[1].base_url is None
    assert models[1].tpm is None


def test_secret_is_resolved_from_environment() -> None:
    models = parse_models(VALID, ENV)

    assert models[0].api_key is not None
    assert models[0].api_key.get_secret_value() == "gsk-secret-value"


def test_secret_never_appears_in_repr_or_str() -> None:
    models = parse_models(VALID, ENV)

    assert "gsk-secret-value" not in repr(models[0])
    assert "gsk-secret-value" not in str(models[0])
    assert "sk-ant-secret" not in repr(models)


def test_key_is_optional_for_local_models() -> None:
    raw = {
        "models": [
            {
                "name": "local",
                "provider": "openai_compatible",
                "model": "m",
                "base_url": "http://localhost:11434/v1",
            }
        ]
    }

    assert parse_models(raw, {})[0].api_key is None


def test_same_alias_may_appear_more_than_once() -> None:
    raw = {"models": [VALID["models"][0], {**VALID["models"][0], "model": "other"}]}  # type: ignore[dict-item]

    assert [m.name for m in parse_models(raw, ENV)] == ["fast", "fast"]


def test_plaintext_key_is_rejected_without_echoing_it() -> None:
    message = problems(with_entry(api_key="sk-live-123456"))

    assert "models[0].api_key" in message
    assert "os.environ/" in message
    assert "sk-live-123456" not in message


def test_missing_environment_variable_is_named() -> None:
    message = problems(VALID, env={"ANTHROPIC_API_KEY": "x"})

    assert "GROQ_API_KEY is not set" in message
    assert "models[0] ('fast')" in message


def test_empty_environment_variable_counts_as_missing() -> None:
    assert "GROQ_API_KEY is not set" in problems(VALID, env={**ENV, "GROQ_API_KEY": ""})


def test_all_problems_are_reported_together() -> None:
    message = problems(VALID, env={})

    assert "GROQ_API_KEY" in message
    assert "ANTHROPIC_API_KEY" in message


def test_unknown_provider_is_rejected() -> None:
    assert "models[0].provider" in problems(with_entry(provider="gemini"))


def test_unknown_field_is_rejected() -> None:
    assert "weigth" in problems(with_entry(weigth=2))


def test_openai_compatible_requires_base_url() -> None:
    assert "base_url is required" in problems(with_entry(base_url=None))


def test_base_url_must_be_http() -> None:
    assert "http" in problems(with_entry(base_url="ftp://x"))


@pytest.mark.parametrize("field", ["weight", "rpm", "tpm"])
def test_limits_must_be_positive(field: str) -> None:
    assert f"models[0].{field}" in problems(with_entry(**{field: 0}))


def test_empty_models_list_is_rejected() -> None:
    assert "models" in problems({"models": []})


def test_missing_models_key_is_rejected() -> None:
    assert "models" in problems({})


def test_load_from_yaml_file(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "models:\n"
        "  - name: fast\n"
        "    provider: openai_compatible\n"
        "    model: m\n"
        "    base_url: https://api.example.com/v1\n"
        "    api_key: os.environ/GROQ_API_KEY\n"
    )

    models = load_models(path, ENV)

    assert models[0].name == "fast"
    assert models[0].api_key is not None


def test_missing_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read model config"):
        load_models(tmp_path / "nope.yaml", ENV)


def test_malformed_yaml_is_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("models: [unclosed")

    with pytest.raises(ConfigError, match="not valid YAML"):
        load_models(path, ENV)


def test_non_mapping_yaml_is_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a list\n")

    with pytest.raises(ConfigError, match="top-level 'models'"):
        load_models(path, ENV)


def test_shipped_example_config_is_valid() -> None:
    models = load_models(
        Path(__file__).parent.parent / "config.example.yaml",
        {"GROQ_API_KEY": "a", "ANTHROPIC_API_KEY": "b"},
    )

    assert {m.name for m in models} == {"fast", "balanced", "smart", "local"}
