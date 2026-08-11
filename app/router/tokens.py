import json
import math

from app.schemas.chat import ChatCompletionRequest, ImagePart, Message, TextPart

# Rough OpenAI-style accounting: every message carries framing tokens, and the reply is primed.
_TOKENS_PER_MESSAGE = 4
_REPLY_PRIMING = 3
_IMAGE_TOKENS_LOW_DETAIL = 85
_IMAGE_TOKENS_DEFAULT = 765  # a typical 1024px image at auto/high detail


def estimate_text_tokens(text: str) -> int:
    """Estimate tokens in ``text`` without a tokenizer.

    English-like text averages about 4 characters per token; characters outside ASCII
    (CJK, emoji, many accented letters) usually cost about a token each. This is an estimate
    for routing, limits and cost previews. Billing always uses the usage the provider reports.
    """
    ascii_chars = len(text.encode("ascii", "ignore"))
    return math.ceil(ascii_chars / 4) + (len(text) - ascii_chars)


def _message_tokens(message: Message) -> int:
    total = _TOKENS_PER_MESSAGE
    content = message.content
    if isinstance(content, str):
        total += estimate_text_tokens(content)
    elif content is not None:
        for part in content:
            if isinstance(part, TextPart):
                total += estimate_text_tokens(part.text)
            elif isinstance(part, ImagePart):
                low = part.image_url.detail == "low"
                total += _IMAGE_TOKENS_LOW_DETAIL if low else _IMAGE_TOKENS_DEFAULT
    if message.name:
        total += 1 + estimate_text_tokens(message.name)
    for call in message.tool_calls or []:
        total += estimate_text_tokens(call.function.name)
        total += estimate_text_tokens(call.function.arguments)
    return total


def estimate_prompt_tokens(request: ChatCompletionRequest) -> int:
    total = _REPLY_PRIMING + sum(_message_tokens(m) for m in request.messages)
    if request.tools:
        schema = json.dumps(
            [t.model_dump(exclude_none=True) for t in request.tools], separators=(",", ":")
        )
        total += estimate_text_tokens(schema)
    return total
