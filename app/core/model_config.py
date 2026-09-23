import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StrictFloat,
    StrictInt,
    ValidationError,
    field_validator,
)

from app.core.tiers import Tier
from app.routing.score import ComplexityWeights

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
    # Total tokens (prompt + completion) the model accepts; omit if unknown or unlimited.
    context_window: int | None = Field(default=None, gt=0)
    # USD per million tokens; 0 means free (local models). Set these from your vendor's price list.
    input_cost_per_mtok: float = Field(default=0.0, ge=0)
    output_cost_per_mtok: float = Field(default=0.0, ge=0)
    # slm / mid / frontier. Needed for smart routing; every entry of one name must agree.
    tier: Tier | None = None

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


class RoutingSection(BaseModel):
    """The optional ``routing:`` block: knobs for the smart router."""

    model_config = ConfigDict(extra="forbid")

    # How strongly each signal argues that a request is hard, 0 to 1 (see app/routing/score.py).
    complexity: dict[str, StrictInt | StrictFloat] = Field(default_factory=dict)


class ModelsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ModelEntry] = Field(min_length=1)
    fallbacks: dict[str, list[str]] = Field(default_factory=dict)
    routing: RoutingSection = Field(default_factory=RoutingSection)


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
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0
    context_window: int | None = None
    tier: Tier | None = None


@dataclass(frozen=True)
class ModelsConfig:
    models: list[ModelConfig]
    # alias -> aliases to try, in order, once every retry on it has failed
    fallbacks: dict[str, tuple[str, ...]]
    # overrides for the complexity signal strengths (validated; see app/routing/score.py)
    complexity: dict[str, float] = field(default_factory=dict)


def _tier_name(tier: Tier | None) -> str:
    return "(none)" if tier is None else f"'{tier.value}'"


def tier_problems(configs: Sequence[ModelConfig]) -> list[str]:
    """Entries that share a name are one model, so they must declare the same tier (or none)."""
    first: dict[str, int] = {}
    problems = []
    for index, config in enumerate(configs):
        earlier = first.setdefault(config.name, index)
        if earlier == index:
            continue
        expected = configs[earlier].tier
        if config.tier != expected:
            problems.append(
                f"models[{index}] ('{config.name}'): tier {_tier_name(config.tier)} conflicts "
                f"with tier {_tier_name(expected)} of models[{earlier}]; entries with the same "
                "name must agree"
            )
    return problems


def _format_location(location: tuple[int | str, ...]) -> str:
    text = ""
    for part in location:
        text += f"[{part}]" if isinstance(part, int) else (f".{part}" if text else str(part))
    return text


def parse_config(raw: object, env: Mapping[str, str] | None = None) -> ModelsConfig:
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
                input_cost_per_mtok=entry.input_cost_per_mtok,
                output_cost_per_mtok=entry.output_cost_per_mtok,
                context_window=entry.context_window,
                tier=entry.tier,
            )
        )

    problems.extend(tier_problems(configs))
    problems.extend(
        f"routing.complexity: {message}"
        for message in ComplexityWeights.problems(parsed.routing.complexity)
    )
    aliases = {config.name for config in configs}
    for source, targets in parsed.fallbacks.items():
        if source not in aliases:
            problems.append(f"fallbacks.{source}: unknown model '{source}'")
        for position, target in enumerate(targets):
            where = f"fallbacks.{source}[{position}]"
            if target not in aliases:
                problems.append(f"{where}: unknown model '{target}'")
            elif target == source:
                problems.append(f"{where}: a model cannot fall back to itself")

    if problems:
        raise ConfigError("invalid model config:\n  - " + "\n  - ".join(problems))
    fallbacks = {source: tuple(targets) for source, targets in parsed.fallbacks.items()}
    complexity = {name: float(value) for name, value in parsed.routing.complexity.items()}
    return ModelsConfig(models=configs, fallbacks=fallbacks, complexity=complexity)


def parse_models(raw: object, env: Mapping[str, str] | None = None) -> list[ModelConfig]:
    return parse_config(raw, env).models


def load_config(path: Path, env: Mapping[str, str] | None = None) -> ModelsConfig:
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
    return parse_config(raw, env)


def load_models(path: Path, env: Mapping[str, str] | None = None) -> list[ModelConfig]:
    return load_config(path, env).models
