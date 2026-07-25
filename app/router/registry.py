from collections.abc import Iterable, Sequence

import httpx

from app.core.model_config import ConfigError, ModelConfig
from app.providers.factory import build_provider
from app.router.deployment import Deployment


class UnknownModelError(Exception):
    def __init__(self, alias: str) -> None:
        super().__init__(f"The model '{alias}' does not exist")
        self.alias = alias


class ModelRegistry:
    """Deployments grouped by the alias clients request, in config order."""

    def __init__(self, deployments: Iterable[Deployment]) -> None:
        by_alias: dict[str, list[Deployment]] = {}
        seen: set[str] = set()
        for deployment in deployments:
            if deployment.id in seen:
                raise ConfigError(
                    f"duplicate deployment for model '{deployment.alias}' "
                    f"({deployment.upstream_model}); identical entries are not allowed"
                )
            seen.add(deployment.id)
            by_alias.setdefault(deployment.alias, []).append(deployment)
        self._by_alias = {alias: tuple(items) for alias, items in by_alias.items()}

    def get(self, alias: str) -> Sequence[Deployment]:
        try:
            return self._by_alias[alias]
        except KeyError:
            raise UnknownModelError(alias) from None

    def aliases(self) -> list[str]:
        return list(self._by_alias)

    def __contains__(self, alias: object) -> bool:
        return alias in self._by_alias

    def __len__(self) -> int:
        return sum(len(items) for items in self._by_alias.values())


def build_registry(configs: Iterable[ModelConfig], client: httpx.AsyncClient) -> ModelRegistry:
    return ModelRegistry(
        Deployment.create(config, build_provider(config, client)) for config in configs
    )
