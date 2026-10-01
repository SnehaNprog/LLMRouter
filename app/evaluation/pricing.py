from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

from app.core.model_config import ModelConfig, load_config
from app.core.tiers import Tier
from app.router.cost import cost_usd


class AnyKey(Mapping[str, str]):
    """An environment in which every variable is set, to a placeholder.

    Prices come from the same config the gateway uses, but pricing never calls a provider, so it
    must not need the real API keys that config refers to.
    """

    def __getitem__(self, key: str) -> str:
        return "unused"

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def get(self, key: str, default: object = None) -> str:
        return "unused"


class PriceBook:
    """What each model alias costs, taken from the same config the gateway runs on.

    An alias with several deployments is priced at their weighted average, which is what its
    traffic costs on average. The evaluation prices every answer from the tokens the gateway
    reports, so the baseline and the router are measured with exactly the same prices.
    """

    def __init__(self, configs: Iterable[ModelConfig]) -> None:
        by_alias: dict[str, list[ModelConfig]] = {}
        for config in configs:
            by_alias.setdefault(config.name, []).append(config)
        self._by_alias = by_alias

    @classmethod
    def from_file(cls, path: Path) -> "PriceBook":
        """The prices in a gateway model config (config errors are raised as ConfigError)."""
        return cls(load_config(path, AnyKey()).models)

    def configs(self) -> list[ModelConfig]:
        return [config for group in self._by_alias.values() for config in group]

    def __contains__(self, alias: object) -> bool:
        return alias in self._by_alias

    def aliases(self) -> list[str]:
        return list(self._by_alias)

    def cost(self, alias: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Dollar cost of an answer from ``alias``; KeyError for an alias that is not configured."""
        deployments = self._by_alias[alias]
        total = sum(d.weight for d in deployments)
        return (
            sum(d.weight * cost_usd(d, prompt_tokens, completion_tokens) for d in deployments)
            / total
        )

    def tier(self, alias: str) -> Tier | None:
        return self._by_alias[alias][0].tier

    def frontier_aliases(self) -> list[str]:
        return [alias for alias in self._by_alias if self.tier(alias) is Tier.FRONTIER]
