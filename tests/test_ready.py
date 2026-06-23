from fastapi.testclient import TestClient

from app.main import app


def test_ready_reports_checks(client: TestClient) -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "checks": {"config": "ok"}}


def test_not_ready_when_a_check_fails(client: TestClient) -> None:
    async def failing() -> bool:
        return False

    original = app.state.readiness_checks
    app.state.readiness_checks = {**original, "db": failing}
    try:
        response = client.get("/ready")
    finally:
        app.state.readiness_checks = original

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["checks"]["db"] == "failed"


def test_not_ready_when_a_check_raises(client: TestClient) -> None:
    async def exploding() -> bool:
        raise ConnectionError("redis down")

    original = app.state.readiness_checks
    app.state.readiness_checks = {"redis": exploding}
    try:
        response = client.get("/ready")
    finally:
        app.state.readiness_checks = original

    assert response.status_code == 503
    assert response.json()["checks"] == {"redis": "failed"}
    assert "redis down" not in response.text


def test_health_is_independent_of_readiness(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
