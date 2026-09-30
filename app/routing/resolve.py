from collections.abc import Callable
from typing import Literal

from app.core.tiers import Tier
from app.router.context import ContextLengthExceededError
from app.router.deployment import Deployment
from app.router.latency import LatencyTracker
from app.router.registry import ModelRegistry, UnknownModelError
from app.routing.policy import AUTO_MODEL
from app.routing.selector import (
    NoRoutableModelError,
    RoutingChoice,
    choose_model,
    pinned_choice,
)
from app.schemas.chat import ChatCompletionRequest


class TenantPolicyError(Exception):
    """An organisation's routing policy cannot be applied right now.

    This is a configuration problem on our side, not something the caller did: the policy points
    at a model that no longer exists, or the cap leaves nothing to route to.
    """

    def __init__(
        self, kind: Literal["pin_unavailable", "no_model_under_cap"], message: str
    ) -> None:
        super().__init__(message)
        self.kind = kind


def resolve_model(
    request: ChatCompletionRequest,
    registry: ModelRegistry,
    latency: LatencyTracker,
    *,
    pinned_model: str | None = None,
    max_tier: Tier | None = None,
    healthy: Callable[[Deployment], bool] | None = None,
) -> tuple[ChatCompletionRequest, RoutingChoice | None]:
    """Turn ``model="auto"`` into a concrete model; any other request passes through untouched.

    The returned request names the chosen model, so everything after this (quotas, retries,
    fallbacks, pricing) treats it exactly as if the caller had asked for that model.

    An organisation's policy shapes the choice: ``pinned_model`` fixes it outright, and
    ``max_tier`` caps how strong a model smart routing may pick. Neither affects requests that
    name a model. ``healthy`` lets smart routing avoid models whose deployments are all down.

    Failures are reported the way a caller would expect them: with no tiered model configured
    "auto" simply does not exist (a 404), and a request too big for every routable model is the
    usual context-length error. A policy that cannot be honoured is a TenantPolicyError.
    """
    if request.model != AUTO_MODEL:
        return request, None
    try:
        if pinned_model is not None:
            choice = _pinned(request, registry, latency, pinned_model, max_tier)
        else:
            choice = choose_model(request, registry, latency, max_tier=max_tier, healthy=healthy)
    except NoRoutableModelError as exc:
        if exc.kind == "context":
            raise ContextLengthExceededError(window=exc.window, required=exc.required) from None
        if exc.kind == "policy":
            raise TenantPolicyError("no_model_under_cap", str(exc)) from None
        raise UnknownModelError(AUTO_MODEL) from None
    return request.model_copy(update={"model": choice.alias}), choice


def _pinned(
    request: ChatCompletionRequest,
    registry: ModelRegistry,
    latency: LatencyTracker,
    alias: str,
    max_tier: Tier | None,
) -> RoutingChoice:
    try:
        tier = registry.tier_of(alias)
    except UnknownModelError:
        raise TenantPolicyError(
            "pin_unavailable", f"the pinned model '{alias}' is not configured"
        ) from None
    if max_tier is not None and tier is not None and tier > max_tier:
        raise TenantPolicyError(
            "pin_unavailable",
            f"the pinned model '{alias}' is tier {tier.value}, above the cap of {max_tier.value}",
        )
    return pinned_choice(request, registry, latency, alias)
