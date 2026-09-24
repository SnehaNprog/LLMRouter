from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from app.core.tiers import Tier
from app.schemas.chat import ChatCompletionRequest

# What a model of each tier is assumed to take, in seconds, until it has been measured.
_LATENCY_PRIORS = {Tier.SLM: 0.6, Tier.MID: 1.5, Tier.FRONTIER: 3.0}
_COMPLETION_CAP = 4_000  # keep one enormous max_tokens from dominating the cost comparison


@dataclass(frozen=True)
class RoutingPolicy:
    """How a complexity score becomes a choice of model. Tunable from the ``routing:`` config."""

    slm_ceiling: float = 0.35  # up to this complexity, a small model is enough
    mid_ceiling: float = 0.65  # up to this, a mid-tier model is; beyond it only a frontier model
    cost_weight: float = 0.7  # how much the price of an answer matters ...
    latency_weight: float = 0.3  # ... against how long it takes
    expected_completion_tokens: int = 300  # assumed answer length when max_tokens is not set
    latency_priors: Mapping[Tier, float] = field(default_factory=lambda: dict(_LATENCY_PRIORS))

    @staticmethod
    def problems(overrides: Mapping[str, Any]) -> list[str]:
        """Everything wrong with a set of overrides, one readable line each."""
        problems = []
        slm = overrides.get("slm_ceiling", RoutingPolicy.slm_ceiling)
        mid = overrides.get("mid_ceiling", RoutingPolicy.mid_ceiling)
        for name in ("slm_ceiling", "mid_ceiling"):
            if name in overrides and not 0 < overrides[name] <= 1:
                problems.append(f"{name}: must be above 0 and at most 1, not {overrides[name]}")
        if not problems and not slm < mid:
            problems.append(f"slm_ceiling ({slm}) must be below mid_ceiling ({mid})")
        for name in ("cost_weight", "latency_weight"):
            if name in overrides and not overrides[name] >= 0:
                problems.append(f"{name}: must not be negative, not {overrides[name]}")
        cost = overrides.get("cost_weight", RoutingPolicy.cost_weight)
        latency = overrides.get("latency_weight", RoutingPolicy.latency_weight)
        if cost + latency <= 0:
            problems.append("cost_weight and latency_weight cannot both be zero")
        tokens = overrides.get("expected_completion_tokens")
        if tokens is not None and tokens < 1:
            problems.append(f"expected_completion_tokens: must be at least 1, not {tokens}")
        return problems

    @classmethod
    def from_overrides(cls, overrides: Mapping[str, Any]) -> "RoutingPolicy":
        """The defaults with some settings replaced."""
        if problems := cls.problems(overrides):
            raise ValueError("invalid routing policy: " + "; ".join(problems))
        return cls(**overrides)

    def required_tier(self, complexity: float) -> Tier:
        """The weakest tier that is still good enough for a request this complex."""
        if complexity <= self.slm_ceiling:
            return Tier.SLM
        if complexity <= self.mid_ceiling:
            return Tier.MID
        return Tier.FRONTIER

    def completion_estimate(self, request: ChatCompletionRequest) -> int:
        """Tokens an answer is expected to run to: what was asked for, up to a sensible cap."""
        per_choice = min(request.max_tokens or self.expected_completion_tokens, _COMPLETION_CAP)
        return per_choice * max(1, request.n)


DEFAULT_POLICY = RoutingPolicy()
