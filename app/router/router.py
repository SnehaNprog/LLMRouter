from collections.abc import AsyncIterator

from app.router.deployment import Deployment
from app.router.registry import ModelRegistry
from app.schemas.chat import ChatCompletionRequest
from app.schemas.completion import ChatCompletionChunk, ChatCompletionResponse


class Router:
    """Sends a request to a deployment of the requested alias.

    Passthrough for now: the first deployment of the alias serves the request.
    Selection, retries and fallbacks are layered on in the following commits.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def select(self, alias: str) -> Deployment:
        return self.registry.get(alias)[0]

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
