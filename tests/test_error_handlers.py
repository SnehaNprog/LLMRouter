from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.api.errors import error_type_for, register_exception_handlers


class Body(BaseModel):
    name: str
    count: int


def make_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.post("/echo")
    def echo(body: Body) -> Body:
        return body

    @app.get("/forbidden")
    def forbidden() -> None:
        raise HTTPException(status_code=403, detail="nope")

    @app.get("/limited")
    def limited() -> None:
        raise HTTPException(status_code=429, detail="slow down", headers={"Retry-After": "7"})

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret internal detail")

    return app


client = TestClient(make_app(), raise_server_exceptions=False)


def test_validation_error_uses_openai_shape() -> None:
    response = client.post("/echo", json={"name": "x", "count": "many"})

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "count"
    assert error["message"].startswith("count:")
    assert error["code"] is None


def test_missing_field_reports_param() -> None:
    response = client.post("/echo", json={"count": 1})

    assert response.json()["error"]["param"] == "name"


def test_http_exception_uses_openai_shape() -> None:
    response = client.get("/forbidden")

    assert response.status_code == 403
    assert response.json()["error"] == {
        "message": "nope",
        "type": "permission_error",
        "param": None,
        "code": None,
    }


def test_http_exception_headers_are_kept() -> None:
    response = client.get("/limited")

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "7"
    assert response.json()["error"]["type"] == "rate_limit_error"


def test_unknown_route_is_openai_shaped_404() -> None:
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found_error"


def test_unhandled_exception_hides_details() -> None:
    response = client.get("/boom")

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["type"] == "api_error"
    assert "secret" not in error["message"]


def test_error_type_mapping() -> None:
    assert error_type_for(401) == "authentication_error"
    assert error_type_for(422) == "invalid_request_error"
    assert error_type_for(503) == "api_error"
