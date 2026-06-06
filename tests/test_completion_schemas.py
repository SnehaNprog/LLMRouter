import pytest
from pydantic import ValidationError

from app.schemas.completion import ChatCompletionChunk, ChatCompletionResponse, Usage

REAL_RESPONSE = {
    "id": "chatcmpl-9abc",
    "object": "chat.completion",
    "created": 1730000000,
    "model": "gpt-4o-mini-2024-07-18",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello! How can I help?"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 9, "completion_tokens": 7, "total_tokens": 16},
}


def test_recorded_response_round_trips() -> None:
    response = ChatCompletionResponse.model_validate(REAL_RESPONSE)

    dumped = response.model_dump(exclude_none=True)

    assert dumped == REAL_RESPONSE


def test_tool_call_response_parses() -> None:
    response = ChatCompletionResponse.model_validate(
        {
            "id": "x",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )

    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls is not None


def test_usage_of_computes_total() -> None:
    assert Usage.of(10, 5).total_tokens == 15


def test_usage_rejects_wrong_total() -> None:
    with pytest.raises(ValidationError):
        Usage(prompt_tokens=1, completion_tokens=2, total_tokens=99)


def test_invalid_finish_reason_is_rejected() -> None:
    bad = {
        "id": "x",
        "created": 1,
        "model": "m",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "oops"}
        ],
    }

    with pytest.raises(ValidationError):
        ChatCompletionResponse.model_validate(bad)


def test_streaming_chunks_parse() -> None:
    first = ChatCompletionChunk.model_validate(
        {
            "id": "c1",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}],
        }
    )
    middle = ChatCompletionChunk.model_validate(
        {
            "id": "c1",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {"content": "Hel"}}],
        }
    )
    last = ChatCompletionChunk.model_validate(
        {
            "id": "c1",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )

    assert first.object == "chat.completion.chunk"
    assert first.choices[0].delta.role == "assistant"
    assert middle.choices[0].delta.content == "Hel"
    assert last.choices[0].finish_reason == "stop"


def test_streaming_tool_call_delta_parses() -> None:
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "c1",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"ci'}}]},
                }
            ],
        }
    )

    delta_calls = chunk.choices[0].delta.tool_calls
    assert delta_calls is not None
    assert delta_calls[0].function is not None
    assert delta_calls[0].function.arguments == '{"ci'
