from collections.abc import Callable
from dataclasses import dataclass, field
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
        kind: Literal["no_tiers", "not_allowed", "context", "policy"],
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
    tier: Tier | None  # None for a model without a tier, which only a pin can select
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
    pinned: bool = False  # the tenant's policy fixed the model; the ranking was not used
    policy: str | None = None  # the tenant policy that shaped this choice, if any
    unavailable: list[str] = field(default_factory=list)  # left out: every deployment is down

    @property
    def tier(self) -> Tier | None:
        return self.candidates[0].tier

    @property
    def escalation(self) -> list[str]:
        """Where to go if the chosen model fails: other candidates, strongest ranking first.

        Models at least as strong as the chosen one come first, in ranked order. Weaker ones that
        still meet the bar come last, as a final resort: a plainer answer beats an error.
        """
        rest = self.candidates[1:]
        own = self.tier
        if own is None:
            return [c.alias for c in rest]
        stronger = [c.alias for c in rest if c.tier is None or c.tier >= own]
        weaker = [c.alias for c in rest if c.tier is not None and c.tier < own]
        return stronger + weaker

    @property
    def reasons(self) -> list[str]:
        """A short, human-readable account of the decision, for logs."""
        top = ", ".join(self.complexity.top(3)) or "nothing in particular"
        tier = self.tier.value if self.tier is not None else "no tier"
        text = [
            f"complexity {self.complexity.value:.2f} (driven by {top}) needs tier "
            f">= {self.required_tier.value}",
            f"chose '{self.alias}' ({tier}) from {len(self.candidates)} candidate(s)",
        ]
        if self.pinned:
            text.append("the organisation's routing policy pins this model")
        elif self.policy:
            text.append(f"the organisation's routing policy applies: {self.policy}")
        if self.unavailable:
            text.append("skipped, circuit breaker open: " + ", ".join(self.unavailable))
        if not self.clears_bar:
            text.append(f"no model of tier {self.required_tier.value} or above was available")
        return text


def _expected_cost(deployments: list[Deployment], prompt: int, completion: int) -> float:
    total_weight = sum(d.weight for d in deployments)
    return (
        sum(d.weight * cost_usd(d.config, prompt, completion) for d in deployments) / total_weight
    )


def _expected_latency(
    deployments: list[Deployment],
    tier: Tier | None,
    latency: LatencyTracker,
    policy: RoutingPolicy,
    fastest_measured: float | None,
) -> tuple[float, bool]:
    """Expected seconds for a model, and whether every one of its deployments has been measured.

    A deployment nobody has timed yet is assumed to take its tier's default, but never longer
    than the fastest deployment that has been timed. Without that cap a model that has not been
    measured could be rated slower than one that has, so it would never be chosen, so it would
    never be measured: starved by its own lack of data. This way it gets tried, and from then on
    it is judged by what it actually does.
    """
    assumed = policy.latency_priors[tier if tier is not None else Tier.MID]
    if fastest_measured is not None:
        assumed = min(assumed, fastest_measured)
    total_weight = sum(d.weight for d in deployments)
    seconds, measured = 0.0, True
    for d in deployments:
        observed = latency.ewma(d.id)
        measured = measured and observed is not None
        seconds += d.weight * (observed if observed is not None else assumed)
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
    max_tier: Tier | None = None,
    healthy: Callable[[Deployment], bool] | None = None,
) -> RoutingChoice:
    """Pick the model for a request: the best value among those strong enough to handle it.

    The request's complexity gives the weakest tier that will do. Every model of that tier or
    stronger is a candidate (so long as the key may use it and its context window fits), and the
    candidates are ranked by expected cost and latency. If nothing is strong enough, the
    strongest models available are used instead and ``clears_bar`` says so.

    ``max_tier`` is an organisation's cap: models stronger than it are never considered, so a
    request that needs more is served by the best model within the cap (``clears_bar`` False).

    ``healthy`` says whether a deployment can be called right now (its circuit breakers are not
    open). A model is judged on its healthy deployments only, and a model with none is left out,
    so traffic goes to whatever is up instead of to something that will only refuse it. If
    nothing at all is healthy the check is dropped: the ranking is as it would be anyway, and the
    request meets the outage honestly further on.

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
    if max_tier is not None:
        within_cap = [(a, t) for a, t in tiered if t <= max_tier]
        if not within_cap:
            raise NoRoutableModelError(
                f"no routable model is at or below the tier {max_tier.value} that this "
                "organisation's routing policy allows",
                "policy",
            )
        tiered = within_cap
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

    def deployments_of(alias: str) -> list[Deployment]:
        return [
            d
            for d in registry.get(alias)
            if fits(d, prompt + completion) and (healthy is None or healthy(d))
        ]

    skipped: list[str] = []
    if healthy is not None:
        up = [(a, t) for a, t in usable if deployments_of(a)]
        if up:
            skipped = [a for a, _ in usable if all(a != b for b, _ in up)]
            usable = up
        else:
            healthy = None  # everything is down: rank as if nothing were

    eligible = [(a, t) for a, t in usable if t >= required]
    clears_bar = bool(eligible)
    if not eligible:  # nothing is strong enough: do the best that can be done
        strongest = max(t for _, t in usable)
        eligible = [(a, t) for a, t in usable if t == strongest]

    observed_seconds = [
        seconds
        for alias, _ in eligible
        for d in registry.get(alias)
        if (healthy is None or healthy(d)) and (seconds := latency.ewma(d.id)) is not None
    ]
    fastest_measured = min(observed_seconds, default=None)

    raw = []
    for position, (alias, tier) in enumerate(eligible):
        deployments = deployments_of(alias)
        cost = _expected_cost(deployments, prompt, completion)
        seconds, measured = _expected_latency(deployments, tier, latency, policy, fastest_measured)
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
                (round(utility, 6), tier.rank if tier is not None else -1, position),
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
        policy=None if max_tier is None else f"max-tier:{max_tier.value}",
        unavailable=skipped,
    )


def pinned_choice(
    request: ChatCompletionRequest,
    registry: ModelRegistry,
    latency: LatencyTracker,
    alias: str,
    *,
    policy: RoutingPolicy | None = None,
    weights: ComplexityWeights | None = None,
) -> RoutingChoice:
    """The choice for an organisation whose routing policy fixes the model.

    Nothing is ranked: the pinned model is used whatever the request's complexity. The score is
    still computed so the decision can be explained, and ``clears_bar`` is False when the pinned
    model is weaker than the request would normally have needed.

    Raises UnknownModelError if the alias is not configured, and NoRoutableModelError if the
    request does not fit its context window.
    """
    policy = policy or registry.routing_policy
    complexity = score_request(request, weights or registry.complexity_weights)
    required = policy.required_tier(complexity.value)
    prompt = estimate_prompt_tokens(request)
    completion = policy.completion_estimate(request)

    available = registry.get(alias)
    usable = [d for d in available if fits(d, prompt + completion)]
    if not usable:
        needed = prompt + completion
        raise NoRoutableModelError(
            f"the request (about {needed} tokens) does not fit the pinned model's context window",
            "context",
            window=max(d.context_window or 0 for d in available),
            required=needed,
        )
    tier = registry.tier_of(alias)
    seconds, measured = _expected_latency(usable, tier, latency, policy, None)
    candidate = Candidate(
        alias=alias,
        tier=tier,
        expected_cost_usd=_expected_cost(usable, prompt, completion),
        expected_latency_s=round(seconds, 4),
        latency_measured=measured,
        utility=0.0,
    )
    return RoutingChoice(
        alias=alias,
        complexity=complexity,
        required_tier=required,
        clears_bar=tier is None or tier >= required,
        candidates=[candidate],
        pinned=True,
        policy="pinned",
    )
