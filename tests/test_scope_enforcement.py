import pytest
from fastapi import FastAPI
from openai import OpenAI, PermissionDeniedError
from sqlalchemy import select, update

from app.auth.dependencies import require_scope
from app.auth.service import revoke_api_key
from app.db.models import ApiKey
from tests.conftest import AuthedClient

BODY = {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]}


def chat(client: AuthedClient, secret: str, body: dict[str, object] | None = None) -> int:
    response = client.post("/v1/chat/completions", json=body or BODY, headers=client.as_key(secret))
    return response.status_code


def models(client: AuthedClient, secret: str) -> int:
    return client.get("/v1/models", headers=client.as_key(secret)).status_code


# ---------- who may do what ----------


def test_default_key_can_chat_and_list_models(client: AuthedClient) -> None:
    assert chat(client, client.secret) == 200
    assert models(client, client.secret) == 200


def test_models_only_key_cannot_chat(client: AuthedClient) -> None:
    secret = client.issue_key(["models:read"])

    assert models(client, secret) == 200
    assert chat(client, secret) == 403


def test_chat_only_key_cannot_list_models(client: AuthedClient) -> None:
    secret = client.issue_key(["chat:write"])

    assert chat(client, secret) == 200
    assert models(client, secret) == 403


def test_resource_wildcard_grants_its_actions(client: AuthedClient) -> None:
    secret = client.issue_key(["chat:*"])

    assert chat(client, secret) == 200
    assert models(client, secret) == 403


def test_admin_keys_cannot_use_the_model_endpoints(client: AuthedClient) -> None:
    secret = client.issue_key(["admin:*"])

    assert chat(client, secret) == 403
    assert models(client, secret) == 403


def test_all_scopes_together(client: AuthedClient) -> None:
    secret = client.issue_key(["chat:write", "models:read", "admin:keys"])

    assert chat(client, secret) == 200
    assert models(client, secret) == 200


# ---------- the response ----------


def test_denial_is_an_openai_shaped_403(client: AuthedClient) -> None:
    secret = client.issue_key(["models:read"])

    response = client.post("/v1/chat/completions", json=BODY, headers=client.as_key(secret))

    assert response.status_code == 403
    assert response.json()["error"] == {
        "message": "This API key does not have the required scope 'chat:write'.",
        "type": "permission_error",
        "param": None,
        "code": "insufficient_scope",
    }


def test_denial_names_the_missing_scope_in_the_header(client: AuthedClient) -> None:
    secret = client.issue_key(["models:read"])

    response = client.post("/v1/chat/completions", json=BODY, headers=client.as_key(secret))

    assert response.headers["WWW-Authenticate"] == (
        'Bearer error="insufficient_scope", scope="chat:write"'
    )


def test_denial_does_not_leak_the_key_or_its_scopes(client: AuthedClient) -> None:
    secret = client.issue_key(["models:read", "admin:tenants"])

    response = client.post("/v1/chat/completions", json=BODY, headers=client.as_key(secret))

    assert secret not in response.text
    assert "admin:tenants" not in response.text
    assert "models:read" not in response.text


# ---------- ordering ----------


def test_missing_credentials_are_still_a_401_not_a_403(client: AuthedClient) -> None:
    response = client.get("/v1/models", headers={"Authorization": ""})

    assert response.status_code == 401


def test_invalid_key_is_a_401(client: AuthedClient) -> None:
    assert models(client, "llmr_" + "x" * 43) == 401


def test_revoked_key_is_a_401_whatever_its_scopes(client: AuthedClient, app_: FastAPI) -> None:
    secret = client.issue_key(["models:read"])

    async def revoke() -> None:
        async with app_.state.db_session_factory() as session:
            key: ApiKey = (
                await session.execute(select(ApiKey).where(ApiKey.key_prefix == secret[:12]))
            ).scalar_one()
            await revoke_api_key(session, key.tenant_id, key.id)

    assert client.portal is not None
    client.portal.call(revoke)

    assert chat(client, secret) == 401  # not 403: authentication fails first


def test_scope_is_checked_before_the_body_is_validated(client: AuthedClient) -> None:
    secret = client.issue_key(["models:read"])

    status = chat(client, secret, {"nonsense": True})

    assert status == 403


def test_scope_is_checked_before_the_model_is_looked_up(client: AuthedClient) -> None:
    secret = client.issue_key(["models:read"])

    status = chat(client, secret, {**BODY, "model": "does-not-exist"})

    assert status == 403  # an unauthorised caller learns nothing about which models exist


def test_streaming_requests_are_denied_with_a_plain_403(client: AuthedClient) -> None:
    secret = client.issue_key(["models:read"])

    response = client.post(
        "/v1/chat/completions", json={**BODY, "stream": True}, headers=client.as_key(secret)
    )

    assert response.status_code == 403
    assert response.headers["content-type"].startswith("application/json")


# ---------- freshness ----------


def test_scope_changes_apply_to_the_next_request(client: AuthedClient, app_: FastAPI) -> None:
    secret = client.issue_key(["chat:write"])
    assert models(client, secret) == 403

    async def grant() -> None:
        async with app_.state.db_session_factory() as session:
            await session.execute(
                update(ApiKey)
                .where(ApiKey.key_prefix == secret[:12])
                .values(scopes=["chat:write", "models:read"])
            )
            await session.commit()

    assert client.portal is not None
    client.portal.call(grant)

    assert models(client, secret) == 200


# ---------- the dependency itself ----------


def test_unknown_scopes_are_rejected_when_the_route_is_defined() -> None:
    with pytest.raises(ValueError, match="not a known scope"):
        require_scope("chat:writ")


def test_require_scope_protects_any_route(client: AuthedClient, app_: FastAPI) -> None:
    from fastapi import Depends

    @app_.get("/admin/ping", dependencies=[Depends(require_scope("admin:keys"))])
    def ping() -> dict[str, bool]:
        return {"ok": True}

    admin = client.issue_key(["admin:keys"])
    wildcard = client.issue_key(["admin:*"])

    assert client.get("/admin/ping").status_code == 403  # the default key
    assert client.get("/admin/ping", headers=client.as_key(admin)).status_code == 200
    assert client.get("/admin/ping", headers=client.as_key(wildcard)).status_code == 200


# ---------- through the OpenAI SDK ----------


def test_sdk_raises_permission_denied(client: AuthedClient) -> None:
    secret = client.issue_key(["models:read"])
    sdk = OpenAI(base_url="http://testserver/v1", api_key=secret, http_client=client)

    with pytest.raises(PermissionDeniedError) as excinfo:
        sdk.chat.completions.create(
            model="test-model", messages=[{"role": "user", "content": "hi"}]
        )

    assert excinfo.value.status_code == 403
    assert excinfo.value.code == "insufficient_scope"


def test_sdk_with_enough_scope_works(client: AuthedClient) -> None:
    secret = client.issue_key(["chat:write", "models:read"])
    sdk = OpenAI(base_url="http://testserver/v1", api_key=secret, http_client=client)

    completion = sdk.chat.completions.create(
        model="test-model", messages=[{"role": "user", "content": "hi"}]
    )

    assert completion.choices[0].message.content
