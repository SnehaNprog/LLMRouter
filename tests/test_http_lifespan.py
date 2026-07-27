from typing import Annotated

import httpx
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.core.http import get_http_client


def test_client_exists_while_running_and_closed_after(app_: FastAPI) -> None:
    with TestClient(app_):
        http_client = app_.state.http_client
        assert isinstance(http_client, httpx.AsyncClient)
        assert not http_client.is_closed

    assert http_client.is_closed


def test_each_startup_gets_a_fresh_client(app_: FastAPI) -> None:
    with TestClient(app_):
        first = app_.state.http_client
    with TestClient(app_):
        second = app_.state.http_client

    assert first is not second


def test_dependency_returns_the_shared_client(app_: FastAPI) -> None:
    @app_.get("/same")
    def same(
        http_client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    ) -> dict[str, bool]:
        return {"shared": http_client is app_.state.http_client}

    with TestClient(app_) as client:
        assert client.get("/same").json() == {"shared": True}
