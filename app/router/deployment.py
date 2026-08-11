import hashlib
import json
from dataclasses import dataclass

from app.core.model_config import ModelConfig
from app.providers.base import Provider
from app.router.cost import cost_usd


def deployment_id(config: ModelConfig) -> str:
    """Stable identifier for one upstream target.

    Built from what identifies the target (alias, provider kind, model, URL and which
    environment variable holds the key), so the same config always gives the same ID.
    Tunables such as weight or rpm are left out, so changing them does not reset any
    per-deployment state (latency stats, circuit breakers). The secret itself is never used.
    """
    identity = json.dumps(
        [config.name, config.provider, config.model, config.base_url, config.api_key_env],
        separators=(",", ":"),
    )
    return "dep_" + hashlib.sha256(identity.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Deployment:
    id: str
    config: ModelConfig
    provider: Provider

    @classmethod
    def create(cls, config: ModelConfig, provider: Provider) -> "Deployment":
        return cls(id=deployment_id(config), config=config, provider=provider)

    @property
    def alias(self) -> str:
        return self.config.name

    @property
    def upstream_model(self) -> str:
        return self.config.model

    @property
    def weight(self) -> int:
        return self.config.weight

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return cost_usd(self.config, prompt_tokens, completion_tokens)
