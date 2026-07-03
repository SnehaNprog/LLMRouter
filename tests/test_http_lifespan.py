from typing import Annotated

import httpx
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.core.http import get_http_client, lifespan
from app.main import app


def test_client_exists_while_running_and_closed_after() -> None:
    with TestClient(app):
        http_client = app.state.http_client
        assert isinstance(http_client, httpx.AsyncClient)
        assert not http_client.is_closed

    assert http_client.is_closed


def test_each_startup_gets_a_fresh_client() -> None:
    with TestClient(app):
        first = app.state.http_client
    with TestClient(app):
        second = app.state.http_client

    assert first is not second


def test_dependency_returns_the_shared_client() -> None:
    probe = FastAPI(lifespan=lifespan)

    @probe.get("/same")
    def same(
        http_client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    ) -> dict[str, bool]:
        return {"shared": http_client is probe.state.http_client}

    with TestClient(probe) as client:
        assert client.get("/same").json() == {"shared": True}
