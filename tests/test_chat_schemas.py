import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatCompletionRequest, Message


def test_sample_openai_payload_parses() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
            "temperature": 0.7,
            "max_tokens": 128,
            "stop": ["\n\n"],
            "stream": True,
            "user": "tenant-1",
        }
    )

    assert request.model == "gpt-4o-mini"
    assert [m.role for m in request.messages] == ["system", "user"]
    assert request.stream is True
    assert request.n == 1


def test_defaults() -> None:
    request = ChatCompletionRequest(model="m", messages=[Message(role="user", content="hi")])

    assert request.stream is False
    assert request.temperature is None
    assert request.max_tokens is None


def test_unknown_fields_are_ignored() -> None:
    request = ChatCompletionRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "logit_bias": {}}
    )

    assert request.model == "m"


def test_invalid_role_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Message.model_validate({"role": "robot", "content": "hi"})


@pytest.mark.parametrize(
    "overrides",
    [
        {"messages": []},
        {"temperature": 2.5},
        {"temperature": -0.1},
        {"top_p": 0},
        {"top_p": 1.5},
        {"max_tokens": 0},
        {"n": 0},
    ],
)
def test_invalid_values_are_rejected(overrides: dict[str, object]) -> None:
    payload: dict[str, object] = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    payload.update(overrides)

    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(payload)


def test_missing_model_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate({"messages": [{"role": "user", "content": "hi"}]})
