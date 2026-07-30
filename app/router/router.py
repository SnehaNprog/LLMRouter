import random
from collections.abc import AsyncIterator

from app.router.deployment import Deployment
from app.router.registry import ModelRegistry
from app.router.selection import weighted_choice
from app.schemas.chat import ChatCompletionRequest
from app.schemas.completion import ChatCompletionChunk, ChatCompletionResponse


class Router:
    """Sends a request to a deployment of the requested alias.

    Deployments are picked at random in proportion to their weight. Retries and
    fallbacks are layered on in the following commits.
    """

    def __init__(self, registry: ModelRegistry, rng: random.Random | None = None) -> None:
        self.registry = registry
        self._rng = rng or random.Random()

    def select(self, alias: str) -> Deployment:
        return weighted_choice(self.registry.get(alias), self._rng)

    @staticmethod
    def _upstream_request(
        request: ChatCompletionRequest, deployment: Deployment
    ) -> ChatCompletionRequest:
        return request.model_copy(update={"model": deployment.upstream_model})

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        deployment = self.select(request.model)
        return await deployment.provider.chat(self._upstream_request(request, deployment))

    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        # Unknown aliases raise here, before any iteration, so the API can answer with a 404.
        deployment = self.select(request.model)
        return deployment.provider.stream(self._upstream_request(request, deployment))
