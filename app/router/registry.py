from collections.abc import Iterable, Mapping, Sequence

import httpx

from app.core.model_config import ConfigError, ModelConfig, tier_problems
from app.core.tiers import TIERS, Tier
from app.providers.factory import build_provider
from app.router.deployment import Deployment
from app.routing.score import DEFAULT_WEIGHTS, ComplexityWeights


class UnknownModelError(Exception):
    def __init__(self, alias: str) -> None:
        super().__init__(f"The model '{alias}' does not exist")
        self.alias = alias


class ModelRegistry:
    """Deployments grouped by the alias clients request, in config order.

    ``fallbacks`` maps an alias to the aliases to try, in order, after it has failed.
    """

    def __init__(
        self,
        deployments: Iterable[Deployment],
        fallbacks: Mapping[str, Sequence[str]] | None = None,
        complexity_weights: ComplexityWeights | None = None,
    ) -> None:
        self.complexity_weights = complexity_weights or DEFAULT_WEIGHTS
        deployments = list(deployments)
        if problems := tier_problems([d.config for d in deployments]):
            raise ConfigError("inconsistent tiers:\n  - " + "\n  - ".join(problems))
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
        self._fallbacks = {alias: tuple(targets) for alias, targets in (fallbacks or {}).items()}
        for source, targets in self._fallbacks.items():
            for name in (source, *targets):
                if name not in self._by_alias:
                    raise ConfigError(f"fallbacks refer to unknown model '{name}'")

    def get(self, alias: str) -> Sequence[Deployment]:
        try:
            return self._by_alias[alias]
        except KeyError:
            raise UnknownModelError(alias) from None

    def chain(self, alias: str) -> list[str]:
        """The alias followed by its fallbacks, depth first, each alias at most once.

        Fallbacks apply transitively: with a -> [b] and b -> [c], the chain for a is
        [a, b, c]. Visiting each alias once makes cycles harmless.
        """
        self.get(alias)  # raises UnknownModelError for aliases that do not exist
        order: list[str] = []

        def visit(name: str) -> None:
            if name in order:
                return
            order.append(name)
            for target in self._fallbacks.get(name, ()):
                visit(target)

        visit(alias)
        return order

    def tier_of(self, alias: str) -> Tier | None:
        """The tier a model was declared with, or None if it takes no part in smart routing."""
        return self.get(alias)[0].config.tier

    def aliases_in_tier(self, tier: Tier) -> list[str]:
        """The models of a tier, in config order."""
        return [alias for alias in self._by_alias if self.tier_of(alias) is tier]

    def tiers(self) -> dict[Tier, list[str]]:
        """Every tier that has models, weakest first, with its models in config order."""
        grouped = {tier: self.aliases_in_tier(tier) for tier in TIERS}
        return {tier: aliases for tier, aliases in grouped.items() if aliases}

    def aliases(self) -> list[str]:
        return list(self._by_alias)

    def __contains__(self, alias: object) -> bool:
        return alias in self._by_alias

    def __len__(self) -> int:
        return sum(len(items) for items in self._by_alias.values())


def build_registry(
    configs: Iterable[ModelConfig],
    client: httpx.AsyncClient,
    fallbacks: Mapping[str, Sequence[str]] | None = None,
    complexity_weights: ComplexityWeights | None = None,
) -> ModelRegistry:
    return ModelRegistry(
        (Deployment.create(config, build_provider(config, client)) for config in configs),
        fallbacks,
        complexity_weights,
    )
