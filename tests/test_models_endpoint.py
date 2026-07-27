from fastapi.testclient import TestClient

from app.schemas.models import ModelList


def test_models_list_matches_openai_shape(client: TestClient) -> None:
    response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    for entry in body["data"]:
        assert set(entry) == {"id", "object", "created", "owned_by"}
        assert entry["object"] == "model"


def test_models_list_comes_from_the_registry(client: TestClient) -> None:
    parsed = ModelList.model_validate(client.get("/v1/models").json())

    assert [m.id for m in parsed.data] == ["test-model"]
