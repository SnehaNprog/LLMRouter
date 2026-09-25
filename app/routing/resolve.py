from app.router.context import ContextLengthExceededError
from app.router.latency import LatencyTracker
from app.router.registry import ModelRegistry, UnknownModelError
from app.routing.policy import AUTO_MODEL
from app.routing.selector import NoRoutableModelError, RoutingChoice, choose_model
from app.schemas.chat import ChatCompletionRequest


def resolve_model(
    request: ChatCompletionRequest, registry: ModelRegistry, latency: LatencyTracker
) -> tuple[ChatCompletionRequest, RoutingChoice | None]:
    """Turn ``model="auto"`` into a concrete model; any other request passes through untouched.

    The returned request names the chosen model, so everything after this (quotas, retries,
    fallbacks, pricing) treats it exactly as if the caller had asked for that model.

    Failures are reported the way a caller would expect them: with no tiered model configured
    "auto" simply does not exist (a 404), and a request too big for every routable model is the
    usual context-length error.
    """
    if request.model != AUTO_MODEL:
        return request, None
    try:
        choice = choose_model(request, registry, latency)
    except NoRoutableModelError as exc:
        if exc.kind == "context":
            raise ContextLengthExceededError(window=exc.window, required=exc.required) from None
        raise UnknownModelError(AUTO_MODEL) from None
    return request.model_copy(update={"model": choice.alias}), choice
