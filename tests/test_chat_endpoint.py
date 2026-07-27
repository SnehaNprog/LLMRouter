from fastapi.testclient import TestClient

from app.schemas.completion import ChatCompletionResponse

VALID = {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]}


def test_returns_valid_completion(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json=VALID)

    assert response.status_code == 200
    parsed = ChatCompletionResponse.model_validate(response.json())
    assert parsed.object == "chat.completion"
    assert parsed.model == "upstream-model"
    assert parsed.choices[0].message.role == "assistant"
    assert parsed.choices[0].finish_reason == "stop"
    assert parsed.usage is not None


def test_response_omits_null_fields(client: TestClient) -> None:
    message = client.post("/v1/chat/completions", json=VALID).json()["choices"][0]["message"]

    assert "tool_calls" not in message


def test_empty_messages_is_openai_shaped_400(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json={"model": "test-model", "messages": []})

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["param"] == "messages"


def test_missing_model_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]}
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "model"
