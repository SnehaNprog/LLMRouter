from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from app.auth.scopes import can_grant
from tests.conftest import AuthedClient

ADMIN = ["admin:tenants", "admin:keys"]


def admin_headers(client: AuthedClient, scopes: list[str] | None = None) -> dict[str, str]:
    return client.as_key(client.issue_key(scopes or ADMIN))


def make_tenant(
    client: AuthedClient, headers: dict[str, str], name: str = "globex"
) -> dict[str, Any]:
    response = client.post("/admin/tenants", json={"name": name}, headers=headers)
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def make_key(
    client: AuthedClient, headers: dict[str, str], tenant_id: str, **body: Any
) -> dict[str, Any]:
    response = client.post(
        f"/admin/tenants/{tenant_id}/keys", json={"name": "ci", **body}, headers=headers
    )
    assert response.status_code == 201, response.text
    created: dict[str, Any] = response.json()
    return created


def error(response: httpx.Response) -> dict[str, Any]:
    body: dict[str, Any] = response.json()["error"]
    return body


# ---------- who may use the admin API ----------


def test_admin_api_requires_authentication(client: AuthedClient) -> None:
    response = client.get("/admin/tenants", headers={"Authorization": ""})

    assert response.status_code == 401


def test_ordinary_keys_cannot_use_the_admin_api(client: AuthedClient) -> None:
    for path in ("/admin/tenants", "/admin/tenants/x/keys"):
        response = client.get(path)  # the default key holds chat and model scopes only

        assert response.status_code == 403
        assert error(response)["code"] == "insufficient_scope"


def test_tenant_admins_cannot_manage_keys_and_vice_versa(client: AuthedClient) -> None:
    tenants_only = admin_headers(client, ["admin:tenants"])
    keys_only = admin_headers(client, ["admin:keys"])
    tenant = make_tenant(client, tenants_only)

    assert client.get("/admin/tenants", headers=keys_only).status_code == 403
    assert (
        client.get(f"/admin/tenants/{tenant['id']}/keys", headers=tenants_only).status_code == 403
    )
    assert client.get(f"/admin/tenants/{tenant['id']}/keys", headers=keys_only).status_code == 200


def test_the_admin_wildcard_covers_both(client: AuthedClient) -> None:
    headers = admin_headers(client, ["admin:*"])
    tenant = make_tenant(client, headers)

    assert client.get(f"/admin/tenants/{tenant['id']}/keys", headers=headers).status_code == 200


def test_admin_keys_cannot_call_models(client: AuthedClient) -> None:
    headers = admin_headers(client)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        headers=headers,
    )

    assert response.status_code == 403


# ---------- tenants ----------


def test_create_tenant(client: AuthedClient) -> None:
    headers = admin_headers(client)

    response = client.post("/admin/tenants", json={"name": "  Globex  "}, headers=headers)

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Globex"
    assert body["is_active"] is True
    assert len(body["id"]) == 32
    assert set(body) == {
        "id",
        "name",
        "is_active",
        "monthly_budget_usd",
        "pinned_model",
        "max_tier",
        "created_at",
    }


def test_duplicate_tenant_names_conflict(client: AuthedClient) -> None:
    headers = admin_headers(client)
    make_tenant(client, headers, "globex")

    response = client.post("/admin/tenants", json={"name": "globex"}, headers=headers)

    assert response.status_code == 409
    assert error(response)["code"] == "tenant_exists"


@pytest.mark.parametrize("body", [{}, {"name": ""}, {"name": "x" * 201}, {"name": "a", "oops": 1}])
def test_invalid_tenant_requests_are_rejected(client: AuthedClient, body: dict[str, Any]) -> None:
    response = client.post("/admin/tenants", json=body, headers=admin_headers(client))

    assert response.status_code == 400
    assert error(response)["type"] == "invalid_request_error"


def test_blank_tenant_name_after_trimming_is_rejected(client: AuthedClient) -> None:
    response = client.post("/admin/tenants", json={"name": "   "}, headers=admin_headers(client))

    assert response.status_code == 400


def test_list_get_tenants(client: AuthedClient) -> None:
    headers = admin_headers(client)
    created = make_tenant(client, headers, "globex")

    listing = client.get("/admin/tenants", headers=headers).json()
    fetched = client.get(f"/admin/tenants/{created['id']}", headers=headers)

    assert listing["object"] == "list"
    assert {t["name"] for t in listing["data"]} == {"test-tenant", "globex"}
    assert fetched.json() == created


def test_tenant_list_pagination(client: AuthedClient) -> None:
    headers = admin_headers(client)
    for index in range(5):
        make_tenant(client, headers, f"t{index}")

    first = client.get("/admin/tenants?limit=3", headers=headers).json()["data"]
    rest = client.get("/admin/tenants?limit=3&offset=3", headers=headers).json()["data"]

    assert len(first) == 3
    assert len(rest) == 3  # 6 tenants in all, counting the caller's
    assert not {t["id"] for t in first} & {t["id"] for t in rest}


@pytest.mark.parametrize("query", ["limit=0", "limit=201", "offset=-1", "limit=abc"])
def test_bad_pagination_is_rejected(client: AuthedClient, query: str) -> None:
    response = client.get(f"/admin/tenants?{query}", headers=admin_headers(client))

    assert response.status_code == 400


def test_unknown_tenant_is_a_404(client: AuthedClient) -> None:
    headers = admin_headers(client)

    for response in (
        client.get("/admin/tenants/nope", headers=headers),
        client.patch("/admin/tenants/nope", json={"name": "x"}, headers=headers),
        client.delete("/admin/tenants/nope", headers=headers),
    ):
        assert response.status_code == 404
        assert error(response)["code"] == "tenant_not_found"


def test_rename_and_deactivate_a_tenant(client: AuthedClient) -> None:
    headers = admin_headers(client)
    tenant = make_tenant(client, headers)

    renamed = client.patch(
        f"/admin/tenants/{tenant['id']}", json={"name": "initech"}, headers=headers
    )
    off = client.patch(f"/admin/tenants/{tenant['id']}", json={"is_active": False}, headers=headers)

    assert renamed.json()["name"] == "initech"
    assert off.json()["is_active"] is False
    assert off.json()["name"] == "initech"


def test_empty_update_is_rejected(client: AuthedClient) -> None:
    headers = admin_headers(client)
    tenant = make_tenant(client, headers)

    response = client.patch(f"/admin/tenants/{tenant['id']}", json={}, headers=headers)

    assert response.status_code == 400
    assert "nothing to update" in error(response)["message"]


def test_renaming_onto_an_existing_name_conflicts(client: AuthedClient) -> None:
    headers = admin_headers(client)
    make_tenant(client, headers, "globex")
    other = make_tenant(client, headers, "initech")

    response = client.patch(
        f"/admin/tenants/{other['id']}", json={"name": "globex"}, headers=headers
    )

    assert response.status_code == 409


def test_deactivating_a_tenant_blocks_its_keys_and_reactivating_restores_them(
    client: AuthedClient,
) -> None:
    headers = admin_headers(client)
    tenant = make_tenant(client, headers)
    secret = make_key(client, headers, tenant["id"])["secret"]

    def use() -> int:
        return client.get("/v1/models", headers=client.as_key(secret)).status_code

    assert use() == 200
    client.patch(f"/admin/tenants/{tenant['id']}", json={"is_active": False}, headers=headers)
    assert use() == 403
    client.patch(f"/admin/tenants/{tenant['id']}", json={"is_active": True}, headers=headers)
    assert use() == 200


def test_deleting_a_tenant_removes_it_and_its_keys(client: AuthedClient) -> None:
    headers = admin_headers(client)
    tenant = make_tenant(client, headers)
    secret = make_key(client, headers, tenant["id"])["secret"]

    response = client.delete(f"/admin/tenants/{tenant['id']}", headers=headers)

    assert response.status_code == 204
    assert response.content == b""
    assert client.get(f"/admin/tenants/{tenant['id']}", headers=headers).status_code == 404
    assert client.get("/v1/models", headers=client.as_key(secret)).status_code == 401


def test_admins_cannot_lock_themselves_out(client: AuthedClient) -> None:
    headers = admin_headers(client)
    own = client.get("/admin/tenants", headers=headers).json()["data"][0]["id"]

    deactivate = client.patch(f"/admin/tenants/{own}", json={"is_active": False}, headers=headers)
    delete = client.delete(f"/admin/tenants/{own}", headers=headers)
    rename = client.patch(f"/admin/tenants/{own}", json={"name": "renamed"}, headers=headers)

    assert deactivate.status_code == delete.status_code == 409
    assert error(deactivate)["code"] == "cannot_modify_own_tenant"
    assert rename.status_code == 200  # harmless changes are fine


# ---------- keys ----------


def test_create_key_returns_the_secret_once(client: AuthedClient) -> None:
    headers = admin_headers(client)
    tenant = make_tenant(client, headers)

    response = client.post(
        f"/admin/tenants/{tenant['id']}/keys", json={"name": "ci pipeline"}, headers=headers
    )

    assert response.status_code == 201
    assert response.headers["Cache-Control"] == "no-store"
    body = response.json()
    assert body["secret"].startswith("llmr_")
    assert body["secret"].startswith(body["key_prefix"])
    assert body["status"] == "active"
    assert body["scopes"] == ["chat:write", "models:read"]
    assert body["allowed_models"] is None
    assert body["tenant_id"] == tenant["id"]
    assert body["last_used_at"] is None


def test_the_new_key_works(client: AuthedClient) -> None:
    headers = admin_headers(client)
    tenant = make_tenant(client, headers)
    secret = make_key(client, headers, tenant["id"])["secret"]

    response = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        headers=client.as_key(secret),
    )

    assert response.status_code == 200


def test_secrets_are_never_shown_again(client: AuthedClient) -> None:
    headers = admin_headers(client)
    tenant = make_tenant(client, headers)
    key = make_key(client, headers, tenant["id"])

    listing = client.get(f"/admin/tenants/{tenant['id']}/keys", headers=headers)
    single = client.get(f"/admin/tenants/{tenant['id']}/keys/{key['id']}", headers=headers)

    assert key["secret"] not in listing.text
    assert key["secret"] not in single.text
    assert "secret" not in single.json()
    assert "key_hash" not in single.text


def test_create_key_with_options(client: AuthedClient) -> None:
    headers = admin_headers(client)
    tenant = make_tenant(client, headers)
    deadline = (datetime.now(UTC) + timedelta(days=7)).isoformat()

    key = make_key(
        client,
        headers,
        tenant["id"],
        scopes=["models:read"],
        allowed_models=["test-*"],
        expires_at=deadline,
    )

    assert key["scopes"] == ["models:read"]
    assert key["allowed_models"] == ["test-*"]
    assert key["expires_at"] is not None


@pytest.mark.parametrize(
    "body",
    [
        {"scopes": ["root"]},
        {"scopes": []},
        {"allowed_models": []},
        {"expires_at": "2001-01-01T00:00:00+00:00"},
        {"expires_at": "2999-01-01T00:00:00"},  # no timezone
        {"name": ""},
        {"nonsense": True},
    ],
)
def test_invalid_key_requests_are_rejected(client: AuthedClient, body: dict[str, Any]) -> None:
    headers = admin_headers(client)
    tenant = make_tenant(client, headers)

    response = client.post(
        f"/admin/tenants/{tenant['id']}/keys", json={"name": "ci", **body}, headers=headers
    )

    assert response.status_code == 400
    assert error(response)["type"] == "invalid_request_error"


def test_keys_for_unknown_or_inactive_tenants(client: AuthedClient) -> None:
    headers = admin_headers(client)
    tenant = make_tenant(client, headers)
    client.patch(f"/admin/tenants/{tenant['id']}", json={"is_active": False}, headers=headers)

    missing = client.post("/admin/tenants/nope/keys", json={"name": "ci"}, headers=headers)
    inactive = client.post(
        f"/admin/tenants/{tenant['id']}/keys", json={"name": "ci"}, headers=headers
    )

    assert missing.status_code == 404
    assert error(missing)["code"] == "tenant_not_found"
    assert inactive.status_code == 409
    assert error(inactive)["code"] == "tenant_inactive"


def test_keys_are_listed_per_tenant_with_pagination(client: AuthedClient) -> None:
    headers = admin_headers(client)
    mine, theirs = make_tenant(client, headers, "mine"), make_tenant(client, headers, "theirs")
    for index in range(3):
        make_key(client, headers, mine["id"], name=f"k{index}")
    make_key(client, headers, theirs["id"], name="other")

    everything = client.get(f"/admin/tenants/{mine['id']}/keys", headers=headers).json()["data"]
    page = client.get(f"/admin/tenants/{mine['id']}/keys?limit=2", headers=headers).json()["data"]

    assert [k["name"] for k in everything] == ["k0", "k1", "k2"]
    assert len(page) == 2
    assert all(k["tenant_id"] == mine["id"] for k in everything)


def test_a_key_cannot_be_reached_through_another_tenant(client: AuthedClient) -> None:
    headers = admin_headers(client)
    owner, other = make_tenant(client, headers, "owner"), make_tenant(client, headers, "other")
    key = make_key(client, headers, owner["id"])
    wrong = f"/admin/tenants/{other['id']}/keys/{key['id']}"

    for response in (
        client.get(wrong, headers=headers),
        client.post(f"{wrong}/revoke", headers=headers),
        client.post(f"{wrong}/rotate", json={}, headers=headers),
    ):
        assert response.status_code == 404
        assert error(response)["code"] == "key_not_found"
    still = client.get(f"/admin/tenants/{owner['id']}/keys/{key['id']}", headers=headers).json()
    assert still["status"] == "active"


def test_revoking_a_key_stops_it_immediately(client: AuthedClient) -> None:
    headers = admin_headers(client)
    tenant = make_tenant(client, headers)
    key = make_key(client, headers, tenant["id"])
    assert client.get("/v1/models", headers=client.as_key(key["secret"])).status_code == 200

    revoked = client.post(f"/admin/tenants/{tenant['id']}/keys/{key['id']}/revoke", headers=headers)
    again = client.post(f"/admin/tenants/{tenant['id']}/keys/{key['id']}/revoke", headers=headers)

    assert revoked.json()["status"] == "revoked"
    assert again.json()["revoked_at"] == revoked.json()["revoked_at"]
    assert client.get("/v1/models", headers=client.as_key(key["secret"])).status_code == 401


def test_rotating_a_key(client: AuthedClient) -> None:
    headers = admin_headers(client)
    tenant = make_tenant(client, headers)
    old = make_key(client, headers, tenant["id"], scopes=["models:read"], allowed_models=["test-*"])

    response = client.post(
        f"/admin/tenants/{tenant['id']}/keys/{old['id']}/rotate", json={}, headers=headers
    )

    assert response.status_code == 201
    assert response.headers["Cache-Control"] == "no-store"
    new = response.json()
    assert new["id"] != old["id"]
    assert new["secret"] != old["secret"]
    assert new["name"] == old["name"]
    assert new["scopes"] == ["models:read"]
    assert new["allowed_models"] == ["test-*"]
    assert client.get("/v1/models", headers=client.as_key(old["secret"])).status_code == 401
    assert client.get("/v1/models", headers=client.as_key(new["secret"])).status_code == 200


def test_rotation_with_a_grace_period(client: AuthedClient) -> None:
    headers = admin_headers(client)
    tenant = make_tenant(client, headers)
    old = make_key(client, headers, tenant["id"])

    client.post(
        f"/admin/tenants/{tenant['id']}/keys/{old['id']}/rotate",
        json={"grace_seconds": 3600},
        headers=headers,
    )

    assert client.get("/v1/models", headers=client.as_key(old["secret"])).status_code == 200
    fetched = client.get(f"/admin/tenants/{tenant['id']}/keys/{old['id']}", headers=headers).json()
    assert fetched["status"] == "active"
    assert fetched["expires_at"] is not None


def test_rotation_can_lift_or_change_restrictions(client: AuthedClient) -> None:
    headers = admin_headers(client)
    tenant = make_tenant(client, headers)
    first = make_key(client, headers, tenant["id"], allowed_models=["test-*"])
    second = make_key(client, headers, tenant["id"], allowed_models=["test-*"])
    base = f"/admin/tenants/{tenant['id']}/keys"

    lifted = client.post(
        f"{base}/{first['id']}/rotate", json={"allowed_models": None}, headers=headers
    )
    changed = client.post(
        f"{base}/{second['id']}/rotate",
        json={"allowed_models": ["other"], "scopes": ["chat:write"]},
        headers=headers,
    )

    assert lifted.json()["allowed_models"] is None
    assert changed.json()["allowed_models"] == ["other"]
    assert changed.json()["scopes"] == ["chat:write"]


def test_dead_keys_cannot_be_rotated(client: AuthedClient) -> None:
    headers = admin_headers(client)
    tenant = make_tenant(client, headers)
    key = make_key(client, headers, tenant["id"])
    base = f"/admin/tenants/{tenant['id']}/keys/{key['id']}"
    client.post(f"{base}/revoke", headers=headers)

    response = client.post(f"{base}/rotate", json={}, headers=headers)

    assert response.status_code == 409
    assert error(response)["code"] == "key_inactive"


@pytest.mark.parametrize("body", [{"grace_seconds": -1}, {"grace_seconds": 99999999}, {"x": 1}])
def test_invalid_rotation_requests_are_rejected(client: AuthedClient, body: dict[str, Any]) -> None:
    headers = admin_headers(client)
    tenant = make_tenant(client, headers)
    key = make_key(client, headers, tenant["id"])

    response = client.post(
        f"/admin/tenants/{tenant['id']}/keys/{key['id']}/rotate", json=body, headers=headers
    )

    assert response.status_code == 400


# ---------- privilege escalation ----------


def test_can_grant_rules() -> None:
    assert can_grant([], "chat:write")  # ordinary scopes are the administrators' to hand out
    assert can_grant(["admin:keys"], "models:read")
    assert can_grant(["admin:keys"], "admin:keys")
    assert not can_grant(["admin:keys"], "admin:tenants")
    assert not can_grant(["admin:keys"], "admin:*")
    assert can_grant(["admin:*"], "admin:tenants")
    assert can_grant(["admin:*"], "admin:*")
    assert not can_grant(["admin:tenants", "admin:keys"], "admin:*")


def test_a_key_admin_cannot_mint_a_tenant_admin(client: AuthedClient) -> None:
    keys_only = admin_headers(client, ["admin:keys"])
    tenant = make_tenant(client, admin_headers(client))

    for scopes in (["admin:tenants"], ["admin:*"], ["chat:write", "admin:tenants"]):
        response = client.post(
            f"/admin/tenants/{tenant['id']}/keys",
            json={"name": "sneaky", "scopes": scopes},
            headers=keys_only,
        )

        assert response.status_code == 403
        assert error(response)["code"] == "scope_escalation"


def test_admins_can_hand_out_ordinary_and_equal_scopes(client: AuthedClient) -> None:
    keys_only = admin_headers(client, ["admin:keys"])
    tenant = make_tenant(client, admin_headers(client))

    ordinary = make_key(client, keys_only, tenant["id"], scopes=["chat:write", "models:read"])
    equal = make_key(client, keys_only, tenant["id"], scopes=["admin:keys"])

    assert ordinary["scopes"] == ["chat:write", "models:read"]
    assert equal["scopes"] == ["admin:keys"]


def test_the_wildcard_holder_may_grant_the_wildcard(client: AuthedClient) -> None:
    root = admin_headers(client, ["admin:*"])
    tenant = make_tenant(client, root)

    key = make_key(client, root, tenant["id"], scopes=["admin:*"])

    assert key["scopes"] == ["admin:*"]


def test_rotation_cannot_be_used_to_copy_a_broader_key(client: AuthedClient) -> None:
    root = admin_headers(client, ["admin:*"])
    keys_only = admin_headers(client, ["admin:keys"])
    tenant = make_tenant(client, root)
    broad = make_key(client, root, tenant["id"], scopes=["admin:*"])
    base = f"/admin/tenants/{tenant['id']}/keys/{broad['id']}/rotate"

    inherited = client.post(base, json={}, headers=keys_only)
    explicit = client.post(base, json={"scopes": ["admin:tenants"]}, headers=keys_only)
    downgraded = client.post(base, json={"scopes": ["models:read"]}, headers=keys_only)

    assert inherited.status_code == explicit.status_code == 403
    assert error(inherited)["code"] == "scope_escalation"
    assert downgraded.status_code == 201  # narrowing is always fine


def test_failed_escalation_creates_nothing(client: AuthedClient) -> None:
    keys_only = admin_headers(client, ["admin:keys"])
    tenant = make_tenant(client, admin_headers(client))

    client.post(
        f"/admin/tenants/{tenant['id']}/keys",
        json={"name": "sneaky", "scopes": ["admin:*"]},
        headers=keys_only,
    )

    listing = client.get(f"/admin/tenants/{tenant['id']}/keys", headers=keys_only).json()
    assert listing["data"] == []
