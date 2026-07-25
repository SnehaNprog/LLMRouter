import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

ENV_PREFIX = "os.environ/"

ProviderKind = Literal["openai_compatible", "anthropic"]


class ConfigError(Exception):
    """The model config is unreadable or invalid. The message lists every problem found."""


class ModelEntry(BaseModel):
    """One deployment as written in YAML. The api_key is only a reference, never a secret."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    provider: ProviderKind
    model: str = Field(min_length=1)
    base_url: str | None = None
    api_key: str | None = None
    weight: int = Field(default=1, ge=1)
    rpm: int | None = Field(default=None, gt=0)
    tpm: int | None = Field(default=None, gt=0)

    @field_validator("api_key")
    @classmethod
    def _must_be_env_reference(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.startswith(ENV_PREFIX) or value == ENV_PREFIX:
            raise ValueError(
                f"must be an environment reference like '{ENV_PREFIX}GROQ_API_KEY'; "
                "secrets never go in the YAML file"
            )
        return value

    @field_validator("base_url")
    @classmethod
    def _must_be_http_url(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")
        return value

    def check_provider_rules(self) -> str | None:
        if self.provider == "openai_compatible" and self.base_url is None:
            return "base_url is required for provider 'openai_compatible'"
        return None


class ModelsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ModelEntry] = Field(min_length=1)


@dataclass(frozen=True)
class ModelConfig:
    """A validated deployment with its secret resolved. The key is hidden from repr and logs."""

    name: str
    provider: ProviderKind
    model: str
    base_url: str | None
    api_key: SecretStr | None
    weight: int
    rpm: int | None
    tpm: int | None
    api_key_env: str | None = None  # name of the variable, never the secret


def _format_location(location: tuple[int | str, ...]) -> str:
    text = ""
    for part in location:
        text += f"[{part}]" if isinstance(part, int) else (f".{part}" if text else str(part))
    return text


def parse_models(raw: object, env: Mapping[str, str] | None = None) -> list[ModelConfig]:
    """Validate parsed YAML and resolve secret references against ``env``."""
    environment = os.environ if env is None else env
    problems: list[str] = []

    try:
        parsed = ModelsFile.model_validate(raw)
    except ValidationError as exc:
        details = [f"{_format_location(error['loc'])}: {error['msg']}" for error in exc.errors()]
        raise ConfigError("invalid model config:\n  - " + "\n  - ".join(details)) from None

    configs: list[ModelConfig] = []
    for index, entry in enumerate(parsed.models):
        label = f"models[{index}] ('{entry.name}')"
        rule_problem = entry.check_provider_rules()
        if rule_problem:
            problems.append(f"{label}: {rule_problem}")
        secret: SecretStr | None = None
        variable: str | None = None
        if entry.api_key is not None:
            variable = entry.api_key.removeprefix(ENV_PREFIX)
            value = environment.get(variable, "")
            if not value:
                problems.append(f"{label}: environment variable {variable} is not set")
            else:
                secret = SecretStr(value)
        configs.append(
            ModelConfig(
                name=entry.name,
                provider=entry.provider,
                model=entry.model,
                base_url=entry.base_url,
                api_key=secret,
                weight=entry.weight,
                rpm=entry.rpm,
                tpm=entry.tpm,
                api_key_env=variable,
            )
        )

    if problems:
        raise ConfigError("invalid model config:\n  - " + "\n  - ".join(problems))
    return configs


def load_models(path: Path, env: Mapping[str, str] | None = None) -> list[ModelConfig]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read model config {path}: {exc.strerror or exc}") from None
    try:
        raw: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping with a top-level 'models' list")
    return parse_models(raw, env)
