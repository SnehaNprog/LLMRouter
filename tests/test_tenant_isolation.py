"""Tenants must never see, use or affect each other's keys, whatever a caller tries."""

import json
import logging
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest
from fastapi import Depends, FastAPI
from sqlalchemy import func, select

from app.auth.dependencies import AuthContext, authenticate
from app.core.logging import JsonFormatter
from app.db.models import ApiKey, Tenant
from app.providers.fake import FakeProvider
from tests.conftest import AuthedClient

ADMIN = ["admin:tenants", "admin:keys"]


class World:
    """A platform admin plus two customer tenants, each with a normal key of their own."""

    def __init__(self, client: AuthedClient) -> None:
        self.client = client
        self.admin = client.as_key(client.issue_key(ADMIN))
        self.a = self.tenant("acme")
        self.b = self.tenant("globex")
        self.a_key = self.key(self.a["id"], name="acme app")
        self.b_key = self.key(self.b["id"], name="globex app")

    def tenant(self, name: str) -> dict[str, Any]:
        response = self.client.post("/admin/tenants", json={"name": name}, headers=self.admin)
        assert response.status_code == 201, response.text
        body: dict[str, Any] = response.json()
        return body

    def key(self, tenant_id: str, **body: Any) -> dict[str, Any]:
        response = self.client.post(
            f"/admin/tenants/{tenant_id}/keys", json={"name": "k", **body}, headers=self.admin
        )
        assert response.status_code == 201, response.text
        created: dict[str, Any] = response.json()
        return created

    def get(self, path: str, secret: str | None = None, **kwargs: Any) -> httpx.Response:
        headers = self.admin if secret is None else self.client.as_key(secret)
        return self.client.get(path, headers=headers, **kwargs)

    def works(self, key: dict[str, Any]) -> bool:
        return self.get("/v1/models", key["secret"]).status_code == 200


@pytest.fixture
def world(client: AuthedClient) -> World:
    return World(client)


# ---------- who a request is ----------


def test_a_key_acts_only_as_its_own_tenant(
    client: AuthedClient, app_: FastAPI, world: World
) -> None:
    @app_.get("/whoami")
    def whoami(auth: AuthContext = Depends(authenticate)) -> dict[str, str]:  # noqa: B008
        return {"tenant_id": auth.tenant_id, "key_id": auth.key_id}

    as_a = world.get("/whoami", world.a_key["secret"]).json()
    as_b = world.get("/whoami", world.b_key["secret"]).json()

    assert as_a == {"tenant_id": world.a["id"], "key_id": world.a_key["id"]}
    assert as_b == {"tenant_id": world.b["id"], "key_id": world.b_key["id"]}


@pytest.mark.parametrize(
    "spoof",
    [
        {"headers": {"X-Tenant-ID": "SPOOF"}},
        {"headers": {"X-Tenant": "SPOOF", "X-Forwarded-User": "SPOOF"}},
        {"headers": {"X-Request-ID": "SPOOF"}},
        {"params": {"tenant_id": "SPOOF", "tenant": "SPOOF"}},
    ],
)
def test_claims_about_another_tenant_are_ignored(
    client: AuthedClient, app_: FastAPI, world: World, spoof: dict[str, Any]
) -> None:
    @app_.get("/whoami2")
    def whoami(auth: AuthContext = Depends(authenticate)) -> dict[str, str]:  # noqa: B008
        return {"tenant_id": auth.tenant_id}

    headers = {**client.as_key(world.a_key["secret"]), **spoof.get("headers", {})}
    spoof["headers"] = headers
    spoofed = client.get("/whoami2", **spoof)

    assert spoofed.json() == {"tenant_id": world.a["id"]}


def test_tenants_with_similar_names_are_separate_tenants(world: World) -> None:
    upper = world.tenant("ACME")
    padded = world.client.post(
        "/admin/tenants", json={"name": "  acme  "}, headers=world.admin
    )  # trims to "acme", which exists

    assert upper["id"] != world.a["id"]
    assert padded.status_code == 409


def test_ids_are_random_not_sequential(world: World) -> None:
    ids = [world.tenant(f"t{n}")["id"] for n in range(30)]
    ids += [world.key(world.a["id"])["id"] for _ in range(30)]

    assert len(set(ids)) == 60
    assert all(re.fullmatch(r"[0-9a-f]{32}", value) for value in ids)
    assert all(value[12] == "4" for value in ids)  # uuid4: random, nothing to enumerate


# ---------- what a tenant can see ----------


def test_a_tenant_key_cannot_use_the_admin_api_at_all(world: World) -> None:
    paths = [
        ("get", "/admin/tenants"),
        ("get", f"/admin/tenants/{world.a['id']}"),
        ("get", f"/admin/tenants/{world.b['id']}"),
        ("get", f"/admin/tenants/{world.a['id']}/keys"),
        ("get", f"/admin/tenants/{world.b['id']}/keys"),
        ("get", f"/admin/tenants/{world.b['id']}/keys/{world.b_key['id']}"),
        ("post", f"/admin/tenants/{world.b['id']}/keys/{world.b_key['id']}/revoke"),
        ("delete", f"/admin/tenants/{world.b['id']}"),
    ]

    for caller in (world.a_key, world.b_key):
        for method, path in paths:
            response = world.client.request(
                method, path, headers=world.client.as_key(caller["secret"])
            )
            assert response.status_code == 403, (method, path)

    assert world.works(world.a_key) and world.works(world.b_key)


def test_key_listings_never_include_another_tenants_keys(world: World) -> None:
    a_ids = {world.a_key["id"], *(world.key(world.a["id"])["id"] for _ in range(3))}
    b_ids = {world.b_key["id"], *(world.key(world.b["id"])["id"] for _ in range(3))}

    for tenant, own, other in ((world.a, a_ids, b_ids), (world.b, b_ids, a_ids)):
        seen: set[str] = set()
        for offset in range(0, 6, 2):  # walk the pages
            page = world.get(f"/admin/tenants/{tenant['id']}/keys?limit=2&offset={offset}")
            seen |= {k["id"] for k in page.json()["data"]}
        assert seen == own
        assert not seen & other


def test_other_tenants_keys_look_exactly_like_missing_keys(world: World) -> None:
    ghost = "f" * 32
    real_foreign = world.b_key["id"]

    def probe(key_id: str) -> list[tuple[int, str, str]]:
        base = f"/admin/tenants/{world.a['id']}/keys/{key_id}"
        responses = [
            world.client.get(base, headers=world.admin),
            world.client.post(f"{base}/revoke", headers=world.admin),
            world.client.post(f"{base}/rotate", json={}, headers=world.admin),
        ]
        return [
            (r.status_code, r.json()["error"]["code"], r.text.replace(key_id, "<id>"))
            for r in responses
        ]

    assert probe(real_foreign) == probe(ghost)


def test_key_ids_prefixes_and_hashes_are_not_credentials(world: World) -> None:
    key = world.a_key

    for stolen in (key["id"], key["key_prefix"], "llmr_" + key["id"], key["tenant_id"]):
        assert world.get("/v1/models", stolen).status_code == 401


# ---------- what a tenant can affect ----------


def test_cross_tenant_revoke_and_rotate_change_nothing(world: World) -> None:
    base = f"/admin/tenants/{world.a['id']}/keys/{world.b_key['id']}"

    world.client.post(f"{base}/revoke", headers=world.admin)
    world.client.post(f"{base}/rotate", json={}, headers=world.admin)

    assert world.works(world.b_key)
    after = world.get(f"/admin/tenants/{world.b['id']}/keys").json()["data"]
    assert [k["status"] for k in after] == ["active"]  # no rotated copy, no revocation
    assert len(world.get(f"/admin/tenants/{world.a['id']}/keys").json()["data"]) == 1


def test_revoking_one_tenants_key_leaves_the_other_working(world: World) -> None:
    world.client.post(
        f"/admin/tenants/{world.a['id']}/keys/{world.a_key['id']}/revoke", headers=world.admin
    )

    assert not world.works(world.a_key)
    assert world.works(world.b_key)


def test_rotating_one_tenants_key_leaves_the_other_working(world: World) -> None:
    world.client.post(
        f"/admin/tenants/{world.a['id']}/keys/{world.a_key['id']}/rotate",
        json={},
        headers=world.admin,
    )

    assert not world.works(world.a_key)
    assert world.works(world.b_key)


def test_deactivating_one_tenant_leaves_the_other_working(world: World) -> None:
    world.client.patch(
        f"/admin/tenants/{world.a['id']}", json={"is_active": False}, headers=world.admin
    )

    assert world.get("/v1/models", world.a_key["secret"]).status_code == 403
    assert world.works(world.b_key)


def test_deleting_one_tenant_leaves_the_other_untouched(
    client: AuthedClient, app_: FastAPI, world: World
) -> None:
    extra_b = world.key(world.b["id"])

    world.client.delete(f"/admin/tenants/{world.a['id']}", headers=world.admin)

    assert world.get("/v1/models", world.a_key["secret"]).status_code == 401
    assert world.works(world.b_key)
    assert world.works(extra_b)

    async def rows() -> tuple[int, int, int]:
        async with app_.state.db_session_factory() as session:
            a_keys = await session.scalar(
                select(func.count()).select_from(ApiKey).where(ApiKey.tenant_id == world.a["id"])
            )
            b_keys = await session.scalar(
                select(func.count()).select_from(ApiKey).where(ApiKey.tenant_id == world.b["id"])
            )
            orphans = await session.scalar(
                select(func.count())
                .select_from(ApiKey)
                .where(ApiKey.tenant_id.not_in(select(Tenant.id)))
            )
            return a_keys or 0, b_keys or 0, orphans or 0

    assert client.portal is not None
    assert client.portal.call(rows) == (0, 2, 0)  # A's keys gone, B's intact, nothing orphaned


def test_policies_belong_to_keys_not_tenants(world: World) -> None:
    narrow = world.key(world.a["id"], scopes=["models:read"], allowed_models=["nothing-*"])

    body = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}

    def chat(key: dict[str, Any]) -> int:
        return world.client.post(
            "/v1/chat/completions", json=body, headers=world.client.as_key(key["secret"])
        ).status_code

    assert chat(narrow) == 403  # restricted...
    assert chat(world.a_key) == 200  # ...while its sibling in the same tenant is not
    assert chat(world.b_key) == 200  # ...and neither is another tenant's key


# ---------- concurrency ----------


def test_concurrent_requests_from_different_tenants_never_mix(
    client: AuthedClient, world: World, provider: FakeProvider
) -> None:
    provider.latency = 0.02  # real waiting, so requests genuinely interleave
    lines: list[dict[str, Any]] = []
    lock = threading.Lock()

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            data = json.loads(JsonFormatter().format(record))
            with lock:
                lines.append(data)

    access = logging.getLogger("app.access")
    handler = Capture()
    access.addHandler(handler)
    callers = [(world.a, world.a_key), (world.b, world.b_key)]
    body = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}

    def call(index: int) -> tuple[str, str, int, str]:
        tenant, key = callers[index % 2]
        request_id = f"req-{index}"
        response = client.post(
            "/v1/chat/completions",
            json=body,
            headers={**client.as_key(key["secret"]), "X-Request-ID": request_id},
        )
        return request_id, tenant["id"], response.status_code, response.headers["X-Request-ID"]

    try:
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(call, range(48)))
    finally:
        access.removeHandler(handler)

    assert all(status == 200 and echoed == rid for rid, _, status, echoed in results)
    expected = {rid: tenant_id for rid, tenant_id, _, _ in results}
    logged = {line["request_id"]: line["tenant_id"] for line in lines if "request_id" in line}
    assert logged == expected  # every log line carries its own request's tenant
    assert len(lines) == 48


# ---------- a randomized campaign ----------


def test_random_operations_never_cross_tenant_boundaries(world: World) -> None:
    """Fire random key operations at four tenants and check each outcome against a model."""
    rng = random.Random(2026)
    tenants = [world.a, world.b, world.tenant("initech"), world.tenant("hooli")]
    keys: dict[str, dict[str, Any]] = {  # key id -> {tenant, secret, status}
        world.a_key["id"]: {
            "tenant": world.a["id"],
            "secret": world.a_key["secret"],
            "status": "active",
        },
        world.b_key["id"]: {
            "tenant": world.b["id"],
            "secret": world.b_key["secret"],
            "status": "active",
        },
    }

    def add(tenant: dict[str, Any]) -> None:
        created = world.key(tenant["id"])
        keys[created["id"]] = {
            "tenant": tenant["id"],
            "secret": created["secret"],
            "status": "active",
        }

    for tenant in tenants:
        add(tenant)

    for _ in range(80):
        action = rng.choice(["create", "revoke", "rotate", "cross_revoke", "cross_rotate"])
        key_id = rng.choice(list(keys))
        owner = keys[key_id]["tenant"]
        stranger = rng.choice([t["id"] for t in tenants if t["id"] != owner])

        if action == "create":
            add(rng.choice(tenants))
        elif action == "revoke":
            world.client.post(f"/admin/tenants/{owner}/keys/{key_id}/revoke", headers=world.admin)
            keys[key_id]["status"] = "revoked"
        elif action == "rotate":
            response = world.client.post(
                f"/admin/tenants/{owner}/keys/{key_id}/rotate", json={}, headers=world.admin
            )
            if keys[key_id]["status"] == "active":
                assert response.status_code == 201
                new = response.json()
                keys[key_id]["status"] = "revoked"
                keys[new["id"]] = {"tenant": owner, "secret": new["secret"], "status": "active"}
            else:
                assert response.status_code == 409
        else:  # an attempt through the wrong tenant: must be refused and change nothing
            verb = "revoke" if action == "cross_revoke" else "rotate"
            response = world.client.post(
                f"/admin/tenants/{stranger}/keys/{key_id}/{verb}", json={}, headers=world.admin
            )
            assert response.status_code == 404

    for tenant in tenants:
        listed = {
            k["id"]: k["status"]
            for k in world.get(f"/admin/tenants/{tenant['id']}/keys?limit=200").json()["data"]
        }
        expected = {i: k["status"] for i, k in keys.items() if k["tenant"] == tenant["id"]}
        assert listed == expected  # nothing leaked in, nothing went missing

    for info in keys.values():
        status = world.get("/v1/models", info["secret"]).status_code
        assert status == (200 if info["status"] == "active" else 401)
