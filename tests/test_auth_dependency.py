import json
import logging
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from openai import AuthenticationError, OpenAI
from sqlalchemy import select, update

from app.auth.keys import generate_key
from app.core.logging import JsonFormatter
from app.db.models import ApiKey, Tenant
from tests.conftest import AuthedClient

BODY = {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]}


# ---------- rejections ----------


def test_missing_header_is_a_401(client: AuthedClient) -> None:
    response = client.get("/v1/models", headers={"Authorization": ""})

    assert response.status_code == 401
    assert response.json()["error"] == {
        "message": (
            "You didn't provide an API key. Send your key in an Authorization header "
            "using Bearer auth (Authorization: Bearer YOUR_KEY)."
        ),
        "type": "authentication_error",
        "param": None,
        "code": "missing_api_key",
    }
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize("header", ["Bearer", "Bearer   ", "Basic dXNlcjpwYXNz", "justtoken"])
def test_unusable_headers_count_as_missing(client: AuthedClient, header: str) -> None:
    response = client.get("/v1/models", headers={"Authorization": header})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "missing_api_key"


def test_unknown_but_well_formed_key_is_rejected(client: AuthedClient) -> None:
    response = client.get(
        "/v1/models", headers={"Authorization": f"Bearer {generate_key().secret}"}
    )

    assert response.status_code == 401
    error = response.json()["error"]
    assert error["code"] == "invalid_api_key"
    assert error["type"] == "authentication_error"
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_malformed_key_is_rejected(client: AuthedClient) -> None:
    response = client.get("/v1/models", headers={"Authorization": "Bearer sk-not-ours"})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


def test_rejected_key_is_never_echoed_back(client: AuthedClient) -> None:
    presented = "sk-live-super-secret-value-123456"

    response = client.get("/v1/models", headers={"Authorization": f"Bearer {presented}"})

    assert presented not in response.text
    assert "super-secret" not in response.text


def test_modified_key_is_rejected(client: AuthedClient) -> None:
    flipped = client.secret[:-1] + ("A" if client.secret[-1] != "A" else "B")

    response = client.get("/v1/models", headers={"Authorization": f"Bearer {flipped}"})

    assert response.status_code == 401


def test_chat_endpoint_is_protected(client: AuthedClient) -> None:
    response = client.post(
        "/v1/chat/completions", json=BODY, headers={"Authorization": "Bearer nope"}
    )

    assert response.status_code == 401


def test_authentication_comes_before_body_validation(client: AuthedClient) -> None:
    response = client.post(
        "/v1/chat/completions", json={"nonsense": True}, headers={"Authorization": ""}
    )

    assert response.status_code == 401


# ---------- acceptance ----------


def test_valid_key_is_accepted(client: AuthedClient) -> None:
    assert client.get("/v1/models").status_code == 200
    assert client.post("/v1/chat/completions", json=BODY).status_code == 200


@pytest.mark.parametrize("scheme", ["bearer", "BEARER", "Bearer"])
def test_scheme_is_case_insensitive(client: AuthedClient, scheme: str) -> None:
    response = client.get("/v1/models", headers={"Authorization": f"{scheme} {client.secret}"})

    assert response.status_code == 200


def test_surrounding_whitespace_is_tolerated(client: AuthedClient) -> None:
    response = client.get("/v1/models", headers={"Authorization": f"Bearer  {client.secret}  "})

    assert response.status_code == 200


def test_other_errors_still_work_for_authenticated_callers(client: AuthedClient) -> None:
    unknown = client.post("/v1/chat/completions", json={**BODY, "model": "nope"})
    invalid = client.post("/v1/chat/completions", json={"model": "test-model", "messages": []})

    assert unknown.status_code == 404
    assert invalid.status_code == 400


def test_health_and_readiness_stay_open(client: AuthedClient) -> None:
    for path in ("/health", "/ready"):
        assert client.get(path, headers={"Authorization": ""}).status_code == 200


# ---------- tenants ----------


def test_deactivated_tenant_is_forbidden(client: AuthedClient, app_: FastAPI) -> None:
    async def deactivate() -> None:
        async with app_.state.db_session_factory() as session:
            await session.execute(update(Tenant).values(is_active=False))
            await session.commit()

    assert client.portal is not None
    client.portal.call(deactivate)

    response = client.get("/v1/models")

    assert response.status_code == 403
    error = response.json()["error"]
    assert error["code"] == "tenant_deactivated"
    assert error["type"] == "permission_error"


# ---------- last_used_at ----------


def stored_key(client: AuthedClient, app_: FastAPI) -> ApiKey:
    async def load() -> ApiKey:
        async with app_.state.db_session_factory() as session:
            loaded: ApiKey = (await session.execute(select(ApiKey))).scalar_one()
            return loaded

    assert client.portal is not None
    key: ApiKey = client.portal.call(load)
    return key


def test_first_use_records_last_used(client: AuthedClient, app_: FastAPI) -> None:
    assert stored_key(client, app_).last_used_at is None

    client.get("/v1/models")

    used = stored_key(client, app_).last_used_at
    assert used is not None
    assert abs(datetime.now(UTC) - used) < timedelta(seconds=10)


def test_last_used_is_not_rewritten_on_every_request(client: AuthedClient, app_: FastAPI) -> None:
    client.get("/v1/models")
    first = stored_key(client, app_).last_used_at

    for _ in range(3):
        client.get("/v1/models")

    assert stored_key(client, app_).last_used_at == first


def test_last_used_is_refreshed_once_stale(client: AuthedClient, app_: FastAPI) -> None:
    long_ago = datetime.now(UTC) - timedelta(hours=1)

    async def age() -> None:
        async with app_.state.db_session_factory() as session:
            await session.execute(update(ApiKey).values(last_used_at=long_ago))
            await session.commit()

    assert client.portal is not None
    client.portal.call(age)

    client.get("/v1/models")

    refreshed = stored_key(client, app_).last_used_at
    assert refreshed is not None
    assert refreshed > long_ago + timedelta(minutes=30)


def test_failed_authentication_records_nothing(client: AuthedClient, app_: FastAPI) -> None:
    client.get("/v1/models", headers={"Authorization": "Bearer nope"})

    assert stored_key(client, app_).last_used_at is None


# ---------- audit log ----------


def test_access_log_names_the_tenant_and_key(client: AuthedClient) -> None:
    lines: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(JsonFormatter().format(record))

    access = logging.getLogger("app.access")
    handler = Capture()
    access.addHandler(handler)
    try:
        client.get("/v1/models")
        client.get("/v1/models", headers={"Authorization": "Bearer nope"})
    finally:
        access.removeHandler(handler)

    authenticated, rejected = (json.loads(line) for line in lines)
    assert authenticated["tenant_id"]
    assert authenticated["key_id"]
    assert client.secret not in lines[0]
    assert "tenant_id" not in rejected


# ---------- through the OpenAI SDK ----------


def test_sdk_with_a_wrong_key_gets_an_authentication_error(client: AuthedClient) -> None:
    sdk = OpenAI(base_url="http://testserver/v1", api_key="llmr_wrong", http_client=client)

    with pytest.raises(AuthenticationError) as excinfo:
        sdk.models.list()

    assert excinfo.value.status_code == 401
    assert excinfo.value.code == "invalid_api_key"
