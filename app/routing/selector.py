from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from app.core.tiers import Tier
from app.router.context import fits
from app.router.cost import cost_usd
from app.router.deployment import Deployment
from app.router.latency import LatencyTracker
from app.router.registry import ModelRegistry
from app.router.tokens import estimate_prompt_tokens
from app.routing.policy import RoutingPolicy
from app.routing.score import ComplexityScore, ComplexityWeights, score_request
from app.schemas.chat import ChatCompletionRequest


class NoRoutableModelError(Exception):
    """Smart routing has no model it could send this request to."""

    def __init__(
        self,
        message: str,
        kind: Literal["no_tiers", "not_allowed", "context"],
        *,
        window: int = 0,
        required: int = 0,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.window = window  # for "context": the largest window any routable model has
        self.required = required  # for "context": the tokens the request needs


@dataclass(frozen=True)
class Candidate:
    alias: str
    tier: Tier
    expected_cost_usd: float
    expected_latency_s: float
    latency_measured: bool  # False when any of its deployments is still on the tier's default
    utility: float  # lower is better: weighted cost (relative) and latency (absolute), 0 to 1


@dataclass(frozen=True)
class RoutingChoice:
    alias: str  # the model to use
    complexity: ComplexityScore
    required_tier: Tier
    clears_bar: bool  # False if nothing as strong as the request needs was available
    candidates: list[Candidate]  # best first; candidates[0] is the choice

    @property
    def tier(self) -> Tier:
        return self.candidates[0].tier

    @property
    def reasons(self) -> list[str]:
        """A short, human-readable account of the decision, for logs."""
        top = ", ".join(self.complexity.top(3)) or "nothing in particular"
        text = [
            f"complexity {self.complexity.value:.2f} (driven by {top}) needs tier "
            f">= {self.required_tier.value}",
            f"chose '{self.alias}' ({self.tier.value}) from {len(self.candidates)} candidate(s)",
        ]
        if not self.clears_bar:
            text.append(f"no model of tier {self.required_tier.value} or above was available")
        return text


def _expected_cost(deployments: list[Deployment], prompt: int, completion: int) -> float:
    total_weight = sum(d.weight for d in deployments)
    return (
        sum(d.weight * cost_usd(d.config, prompt, completion) for d in deployments) / total_weight
    )


def _expected_latency(
    deployments: list[Deployment], tier: Tier, latency: LatencyTracker, policy: RoutingPolicy
) -> tuple[float, bool]:
    total_weight = sum(d.weight for d in deployments)
    seconds, measured = 0.0, True
    for d in deployments:
        observed = latency.ewma(d.id)
        measured = measured and observed is not None
        seconds += d.weight * (observed if observed is not None else policy.latency_priors[tier])
    return seconds / total_weight, measured


def choose_model(
    request: ChatCompletionRequest,
    registry: ModelRegistry,
    latency: LatencyTracker,
    *,
    policy: RoutingPolicy | None = None,
    weights: ComplexityWeights | None = None,
    allow: Callable[[str], bool] | None = None,
    score: ComplexityScore | None = None,
) -> RoutingChoice:
    """Pick the model for a request: the best value among those strong enough to handle it.

    The request's complexity gives the weakest tier that will do. Every model of that tier or
    stronger is a candidate (so long as the key may use it and its context window fits), and the
    candidates are ranked by expected cost and latency. If nothing is strong enough, the
    strongest models available are used instead and ``clears_bar`` says so.

    Raises NoRoutableModelError if no tiered model can take the request at all.
    """
    policy = policy or registry.routing_policy
    complexity = score or score_request(request, weights or registry.complexity_weights)
    required = policy.required_tier(complexity.value)
    prompt = estimate_prompt_tokens(request)
    completion = policy.completion_estimate(request)

    tiered = [(alias, tier) for tier, names in registry.tiers().items() for alias in names]
    if not tiered:
        raise NoRoutableModelError(
            "no model has a tier, so there is nothing to route between", "no_tiers"
        )
    allowed = [(a, t) for a, t in tiered if allow is None or allow(a)]
    if not allowed:
        raise NoRoutableModelError("this key may not use any of the routable models", "not_allowed")
    usable = [
        (a, t) for a, t in allowed if any(fits(d, prompt + completion) for d in registry.get(a))
    ]
    if not usable:
        needed = prompt + completion
        largest = max(d.context_window or 0 for a, _ in allowed for d in registry.get(a))
        raise NoRoutableModelError(
            f"the request (about {needed} tokens) fits no routable model's context window",
            "context",
            window=largest,
            required=needed,
        )

    eligible = [(a, t) for a, t in usable if t >= required]
    clears_bar = bool(eligible)
    if not eligible:  # nothing is strong enough: do the best that can be done
        strongest = max(t for _, t in usable)
        eligible = [(a, t) for a, t in usable if t == strongest]

    raw = []
    for position, (alias, tier) in enumerate(eligible):
        deployments = [d for d in registry.get(alias) if fits(d, prompt + completion)]
        cost = _expected_cost(deployments, prompt, completion)
        seconds, measured = _expected_latency(deployments, tier, latency, policy)
        raw.append((alias, tier, cost, seconds, measured, position))

    # Price is judged against the dearest candidate: spending twice as much is twice as bad.
    # Speed is judged in absolute terms: a model 0.2 ms quicker than another is not better in any
    # way that matters, however many times quicker that is, so seconds are squashed onto 0 to 1
    # with a curve that is flat near zero and reaches 0.5 at ``latency_scale_seconds``.
    top_cost = max(cost for _, _, cost, _, _, _ in raw)
    weight_sum = policy.cost_weight + policy.latency_weight
    ranked = []
    for alias, tier, cost, seconds, measured, position in raw:
        cost_share = cost / top_cost if top_cost > 0 else 0.0
        latency_share = seconds / (seconds + policy.latency_scale_seconds)
        utility = (
            policy.cost_weight * cost_share + policy.latency_weight * latency_share
        ) / weight_sum
        ranked.append(
            (
                (round(utility, 6), tier.rank, position),
                Candidate(
                    alias=alias,
                    tier=tier,
                    expected_cost_usd=cost,
                    expected_latency_s=round(seconds, 4),
                    latency_measured=measured,
                    utility=round(utility, 6),
                ),
            )
        )
    ordered = [candidate for _, candidate in sorted(ranked, key=lambda pair: pair[0])]
    return RoutingChoice(
        alias=ordered[0].alias,
        complexity=complexity,
        required_tier=required,
        clears_bar=clears_bar,
        candidates=ordered,
    )
