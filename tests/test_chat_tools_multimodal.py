import pytest
from pydantic import ValidationError

from app.schemas.chat import (
    ChatCompletionRequest,
    ImagePart,
    Message,
    TextPart,
    ToolChoiceNamed,
)


def test_multimodal_content_parses() -> None:
    message = Message.model_validate(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "https://x/y.png", "detail": "low"}},
            ],
        }
    )

    assert isinstance(message.content, list)
    assert isinstance(message.content[0], TextPart)
    assert isinstance(message.content[1], ImagePart)
    assert message.content[1].image_url.detail == "low"


def test_string_content_still_parses() -> None:
    assert Message.model_validate({"role": "user", "content": "hi"}).content == "hi"


def test_unknown_content_part_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Message.model_validate({"role": "user", "content": [{"type": "audio", "data": "x"}]})


def test_tool_calling_request_parses() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        }
    )

    assert request.tools is not None
    assert request.tools[0].function.name == "get_weather"
    assert isinstance(request.tool_choice, ToolChoiceNamed)
    assert request.messages[1].tool_calls is not None
    assert request.messages[1].tool_calls[0].function.arguments == '{"city":"Paris"}'


@pytest.mark.parametrize("choice", ["none", "auto", "required"])
def test_string_tool_choice_values(choice: str) -> None:
    request = ChatCompletionRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "tool_choice": choice}
    )

    assert request.tool_choice == choice


def test_invalid_tool_choice_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "tool_choice": "maybe"}
        )


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user"},
        {"role": "tool", "content": "x"},
        {
            "role": "user",
            "content": "x",
            "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": "{}"}}],
        },
    ],
)
def test_invalid_message_shapes_are_rejected(message: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Message.model_validate(message)
