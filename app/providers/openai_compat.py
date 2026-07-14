import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.providers.base import Provider
from app.providers.errors import ProviderDown
from app.providers.http_errors import (
    error_for_response,
    error_for_stream_event,
    translate_transport_errors,
)
from app.providers.sse import iter_sse
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

    def _payload(self, request: ChatCompletionRequest, *, stream: bool) -> dict[str, Any]:
        payload = request.model_dump(mode="json", exclude_none=True)
        payload["stream"] = stream
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        async with translate_transport_errors(self.name):
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=self._payload(request, stream=False),
                headers=self._headers(),
            )
        if response.is_error:
            raise error_for_response(self.name, response)
        try:
            return ChatCompletionResponse.model_validate(response.json())
        except ValueError as exc:  # invalid JSON or a body that is not a completion
            raise ProviderDown(f"{self.name}: unreadable response", provider=self.name) from exc

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        http_request = self._client.build_request(
            "POST",
            f"{self._base_url}/chat/completions",
            json=self._payload(request, stream=True),
            headers=self._headers(),
        )
        async with translate_transport_errors(self.name):
            response = await self._client.send(http_request, stream=True)
        try:
            if response.is_error:
                await response.aread()
                raise error_for_response(self.name, response)
            async with translate_transport_errors(self.name):
                async for event in iter_sse(response):
                    if event.data.strip() == "[DONE]":
                        return
                    yield self._parse_chunk(event.data)
        finally:
            await response.aclose()

    def _parse_chunk(self, data: str) -> ChatCompletionChunk:
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict) and "error" in parsed:
                error = parsed["error"]
                message = error.get("message", "stream error") if isinstance(error, dict) else error
                kind = error.get("type", "") if isinstance(error, dict) else ""
                raise error_for_stream_event(self.name, str(kind), str(message))
            return ChatCompletionChunk.model_validate(parsed)
        except ValueError as exc:
            raise ProviderDown(f"{self.name}: unreadable stream chunk", provider=self.name) from exc
