import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.providers.base import Provider
from app.providers.errors import BadRequest, ProviderDown
from app.providers.http_errors import (
    error_for_response,
    error_for_stream_event,
    translate_transport_errors,
)
from app.providers.sse import iter_sse
from app.schemas.chat import (
    ChatCompletionRequest,
    ContentPart,
    FunctionCall,
    ImagePart,
    Message,
    TextPart,
    ToolCall,
)
from app.schemas.completion import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    Choice,
    ChunkChoice,
    Delta,
    DeltaFunctionCall,
    DeltaToolCall,
    FinishReason,
    ResponseMessage,
    Usage,
)

ANTHROPIC_VERSION = "2023-06-01"

_FINISH_REASONS: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


def _text_of(content: str | list[ContentPart] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(part.text for part in content if isinstance(part, TextPart))


def _image_block(part: ImagePart) -> dict[str, Any]:
    url = part.image_url.url
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = header.removeprefix("data:").split(";")[0]
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _content_blocks(content: str | list[ContentPart] | None) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextPart):
            if part.text:
                blocks.append({"type": "text", "text": part.text})
        else:
            blocks.append(_image_block(part))
    return blocks


def _assistant_blocks(message: Message) -> list[dict[str, Any]]:
    blocks = _content_blocks(message.content)
    for call in message.tool_calls or []:
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except ValueError as exc:
            raise BadRequest(
                f"tool call {call.id} has arguments that are not valid JSON", provider="anthropic"
            ) from exc
        blocks.append(
            {"type": "tool_use", "id": call.id, "name": call.function.name, "input": arguments}
        )
    return blocks


def _tool_choice(choice: Any) -> dict[str, Any]:
    if choice == "none":
        return {"type": "none"}
    if choice == "required":
        return {"type": "any"}
    if choice == "auto" or choice is None:
        return {"type": "auto"}
    return {"type": "tool", "name": choice.function.name}


def translate_request(
    request: ChatCompletionRequest, default_max_tokens: int, *, stream: bool = False
) -> dict[str, Any]:
    """OpenAI chat request -> Anthropic Messages request."""
    if request.n != 1:
        raise BadRequest("anthropic does not support n > 1", provider="anthropic")

    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []

    def add(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": list(blocks)})

    for message in request.messages:
        if message.role == "system":
            text = _text_of(message.content)
            if text:
                system_parts.append(text)
        elif message.role == "tool":
            add(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": _text_of(message.content),
                    }
                ],
            )
        elif message.role == "assistant":
            add("assistant", _assistant_blocks(message))
        else:
            add("user", _content_blocks(message.content))

    payload: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens or default_max_tokens,
        "messages": messages,
        "stream": stream,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    if request.temperature is not None:
        payload["temperature"] = min(request.temperature, 1.0)  # Anthropic caps at 1
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.stop is not None:
        payload["stop_sequences"] = (
            [request.stop] if isinstance(request.stop, str) else request.stop
        )
    if request.tools:
        payload["tools"] = [
            {
                "name": tool.function.name,
                "description": tool.function.description or "",
                "input_schema": tool.function.parameters or {"type": "object", "properties": {}},
            }
            for tool in request.tools
        ]
        if request.tool_choice is not None:
            payload["tool_choice"] = _tool_choice(request.tool_choice)
    return payload


def translate_response(data: dict[str, Any]) -> ChatCompletionResponse:
    """Anthropic Messages response -> OpenAI chat completion."""
    texts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in data["content"]:
        if block["type"] == "text":
            texts.append(block["text"])
        elif block["type"] == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block["id"],
                    function=FunctionCall(
                        name=block["name"], arguments=json.dumps(block.get("input", {}))
                    ),
                )
            )
    content = "".join(texts)
    message = ResponseMessage(
        content=content if content or not tool_calls else None,
        tool_calls=tool_calls or None,
    )
    usage = data["usage"]
    return ChatCompletionResponse(
        id=data["id"],
        created=int(time.time()),
        model=data["model"],
        choices=[
            Choice(
                index=0,
                message=message,
                finish_reason=_FINISH_REASONS.get(data.get("stop_reason") or "", "stop"),
            )
        ],
        usage=Usage.of(usage["input_tokens"], usage["output_tokens"]),
    )


class StreamTranslator:
    """Turns Anthropic's streaming events into OpenAI chunks, one event at a time."""

    def __init__(self) -> None:
        self._id = ""
        self._model = ""
        self._created = int(time.time())
        self._input_tokens = 0
        self._tool_index_by_block: dict[int, int] = {}

    def _chunk(
        self, delta: Delta, finish: FinishReason | None = None, usage: Usage | None = None
    ) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id=self._id,
            created=self._created,
            model=self._model,
            choices=[ChunkChoice(index=0, delta=delta, finish_reason=finish)],
            usage=usage,
        )

    def feed(self, event_type: str, data: dict[str, Any]) -> list[ChatCompletionChunk]:
        if event_type == "message_start":
            message = data["message"]
            self._id = message["id"]
            self._model = message["model"]
            self._input_tokens = message.get("usage", {}).get("input_tokens", 0)
            return [self._chunk(Delta(role="assistant", content=""))]

        if event_type == "content_block_start":
            block = data["content_block"]
            if block["type"] != "tool_use":
                return []
            tool_index = len(self._tool_index_by_block)
            self._tool_index_by_block[data["index"]] = tool_index
            call = DeltaToolCall(
                index=tool_index,
                id=block["id"],
                type="function",
                function=DeltaFunctionCall(name=block["name"], arguments=""),
            )
            return [self._chunk(Delta(tool_calls=[call]))]

        if event_type == "content_block_delta":
            delta = data["delta"]
            if delta["type"] == "text_delta":
                return [self._chunk(Delta(content=delta["text"]))]
            if delta["type"] == "input_json_delta":
                call = DeltaToolCall(
                    index=self._tool_index_by_block[data["index"]],
                    function=DeltaFunctionCall(arguments=delta["partial_json"]),
                )
                return [self._chunk(Delta(tool_calls=[call]))]
            return []  # e.g. thinking deltas

        if event_type == "message_delta":
            reason = _FINISH_REASONS.get(data["delta"].get("stop_reason") or "", "stop")
            output_tokens = data.get("usage", {}).get("output_tokens", 0)
            usage = Usage.of(self._input_tokens, output_tokens)
            return [self._chunk(Delta(), finish=reason, usage=usage)]

        return []  # ping, content_block_stop, message_stop


class AnthropicProvider(Provider):
    def __init__(
        self,
        name: str = "anthropic",
        *,
        api_key: str,
        client: httpx.AsyncClient,
        base_url: str = "https://api.anthropic.com",
        default_max_tokens: int = 4096,
    ) -> None:
        self.name = name
        self._api_key = api_key
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._default_max_tokens = default_max_tokens

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        payload = translate_request(request, self._default_max_tokens)
        async with translate_transport_errors(self.name):
            response = await self._client.post(
                f"{self._base_url}/v1/messages", json=payload, headers=self._headers()
            )
        if response.is_error:
            raise error_for_response(self.name, response)
        try:
            return translate_response(response.json())
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderDown(f"{self.name}: unreadable response", provider=self.name) from exc

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        payload = translate_request(request, self._default_max_tokens, stream=True)
        http_request = self._client.build_request(
            "POST", f"{self._base_url}/v1/messages", json=payload, headers=self._headers()
        )
        async with translate_transport_errors(self.name):
            response = await self._client.send(http_request, stream=True)
        translator = StreamTranslator()
        try:
            if response.is_error:
                await response.aread()
                raise error_for_response(self.name, response)
            async with translate_transport_errors(self.name):
                async for event in iter_sse(response):
                    for chunk in self._handle(translator, event.event, event.data):
                        yield chunk
        finally:
            await response.aclose()

    def _handle(
        self, translator: StreamTranslator, event_type: str | None, raw: str
    ) -> list[ChatCompletionChunk]:
        try:
            data = json.loads(raw)
            kind = event_type or data.get("type", "")
            if kind == "error":
                error = data.get("error", {})
                raise error_for_stream_event(
                    self.name, str(error.get("type", "")), str(error.get("message", "stream error"))
                )
            return translator.feed(kind, data)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ProviderDown(f"{self.name}: unreadable stream event", provider=self.name) from exc
