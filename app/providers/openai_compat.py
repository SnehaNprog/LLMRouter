from collections.abc import AsyncIterator

import httpx

from app.providers.base import Provider
from app.providers.errors import ProviderDown
from app.providers.http_errors import error_for_response, translate_transport_errors
from app.schemas.chat import ChatCompletionRequest
from app.schemas.completion import ChatCompletionChunk, ChatCompletionResponse


class OpenAICompatibleProvider(Provider):
    """Adapter for any API that speaks OpenAI's /chat/completions (OpenAI, Groq, Ollama, vLLM)."""

    def __init__(
        self,
        name: str,
        *,
        base_url: str,
        api_key: str | None,
        client: httpx.AsyncClient,
    ) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        payload = request.model_dump(mode="json", exclude_none=True)
        payload["stream"] = False
        async with translate_transport_errors(self.name):
            response = await self._client.post(
                f"{self._base_url}/chat/completions", json=payload, headers=self._headers()
            )
        if response.is_error:
            raise error_for_response(self.name, response)
        try:
            return ChatCompletionResponse.model_validate(response.json())
        except ValueError as exc:  # invalid JSON or a body that is not a completion
            raise ProviderDown(f"{self.name}: unreadable response", provider=self.name) from exc

    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        raise NotImplementedError("streaming is added in a later commit")
