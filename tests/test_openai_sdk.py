import pytest
from fastapi.testclient import TestClient
from openai import BadRequestError, OpenAI


@pytest.fixture
def sdk(client: TestClient) -> OpenAI:
    # TestClient is an httpx.Client, so the SDK talks to the app in-process.
    return OpenAI(base_url="http://testserver/v1", api_key="test-key", http_client=client)


def test_chat_completion(sdk: OpenAI) -> None:
    completion = sdk.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi"}],
    )

    assert completion.object == "chat.completion"
    assert completion.model == "gpt-4o-mini"
    assert completion.choices[0].message.role == "assistant"
    assert completion.choices[0].message.content
    assert completion.choices[0].finish_reason == "stop"
    assert completion.usage is not None
    assert completion.usage.total_tokens == (
        completion.usage.prompt_tokens + completion.usage.completion_tokens
    )


def test_chat_completion_with_tools_and_multimodal(sdk: OpenAI) -> None:
    completion = sdk.chat.completions.create(
        model="m",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                ],
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {"name": "f", "parameters": {"type": "object", "properties": {}}},
            }
        ],
        tool_choice="auto",
        temperature=0.2,
    )

    assert completion.choices[0].message.content


def test_list_models(sdk: OpenAI) -> None:
    models = sdk.models.list()

    assert "auto" in [m.id for m in models.data]


def test_bad_request_raises_sdk_error(sdk: OpenAI) -> None:
    with pytest.raises(BadRequestError) as excinfo:
        sdk.chat.completions.create(model="m", messages=[])

    assert excinfo.value.status_code == 400
    assert excinfo.value.type == "invalid_request_error"
    assert excinfo.value.param == "messages"
