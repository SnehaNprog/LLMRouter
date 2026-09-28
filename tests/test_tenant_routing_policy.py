import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pytest
from alembic import command
from sqlalchemy import update

from app.auth.tenants import create_tenant, update_tenant
from app.core.config import Settings
from app.core.tiers import Tier
from app.db.models import Tenant
from app.main import create_app
from app.providers.fake import FakeProvider
from app.router.deployment import Deployment
from app.router.latency import LatencyTracker
from app.router.registry import ModelRegistry, UnknownModelError
from app.routing.resolve import TenantPolicyError, resolve_model
from app.routing.selector import NoRoutableModelError, choose_model, pinned_choice
from app.schemas.chat import ChatCompletionRequest
from tests.conftest import AuthedClient, model_config
from tests.test_auto_routing import EASY, HARD, MEDIUM, answer, ask
from tests.test_db import alembic_config

SPEC: dict[str, tuple[Tier | None, float, float]] = {
    "cheap": (Tier.SLM, 0.05, 0.08),
    "middle": (Tier.MID, 0.6, 0.8),
    "top": (Tier.FRONTIER, 3.0, 15.0),
}


def build(
    spec: dict[str, tuple[Tier | None, float, float]] | None = None,
    *,
    fail: dict[str, int] | None = None,
    fallbacks: dict[str, list[str]] | None = None,
) -> tuple[AuthedClient, dict[str, FakeProvider]]:
    spec = spec or SPEC
    providers = {
        name: FakeProvider(name, reply=f"answer from {name}", fail_first=(fail or {}).get(name, 0))
        for name in spec
    }

    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return ModelRegistry(
            [
                Deployment.create(
                    model_config(
                        name=name,
                        model=f"{name}-up",
                        tier=tier,
                        input_cost_per_mtok=price_in,
                        output_cost_per_mtok=price_out,
                    ),
                    providers[name],
                )
                for name, (tier, price_in, price_out) in spec.items()
            ],
            fallbacks,
        )

    settings = Settings(_env_file=None, retry_base_delay=0.0, retry_max_attempts=1)
    return AuthedClient(create_app(settings, factory)), providers


def admin(client: AuthedClient) -> dict[str, str]:
    return client.as_key(client.issue_key(["admin:tenants", "admin:keys"]))


def organisation(client: AuthedClient, name: str, **policy: Any) -> tuple[str, str]:
    """Create a tenant with the given routing policy and a key; returns (tenant id, key secret)."""
    headers = admin(client)
    created = client.post("/admin/tenants", json={"name": name, **policy}, headers=headers)
    assert created.status_code == 201, created.text
    tenant_id = str(created.json()["id"])
    key = client.post(
        f"/admin/tenants/{tenant_id}/keys", json={"name": "k"}, headers=headers
    ).json()
    return tenant_id, str(key["secret"])


def policy_headers(response: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in response.headers.items() if k.startswith("x-llmrouter-")}


# ---------- setting a policy ----------


def test_a_tenant_can_be_created_with_a_policy() -> None:
    client, _ = build()
    with client:
        headers = admin(client)
        created = client.post(
            "/admin/tenants",
            json={"name": "acme", "pinned_model": "middle", "max_tier": "frontier"},
            headers=headers,
        ).json()
        plain = client.post("/admin/tenants", json={"name": "plain"}, headers=headers).json()

    assert (created["pinned_model"], created["max_tier"]) == ("middle", "frontier")
    assert (plain["pinned_model"], plain["max_tier"]) == (None, None)


def test_a_policy_can_be_set_changed_and_cleared() -> None:
    client, _ = build()
    with client:
        headers = admin(client)
        tenant = client.post("/admin/tenants", json={"name": "acme"}, headers=headers).json()["id"]
        url = f"/admin/tenants/{tenant}"

        pinned = client.patch(url, json={"pinned_model": "cheap"}, headers=headers).json()
        capped = client.patch(url, json={"max_tier": "mid"}, headers=headers).json()
        renamed = client.patch(url, json={"name": "renamed"}, headers=headers).json()
        unpinned = client.patch(url, json={"pinned_model": None}, headers=headers).json()
        uncapped = client.patch(url, json={"max_tier": None}, headers=headers).json()

    assert (pinned["pinned_model"], pinned["max_tier"]) == ("cheap", None)
    assert (capped["pinned_model"], capped["max_tier"]) == (
        "cheap",
        "mid",
    )  # one change keeps the other
    assert (renamed["pinned_model"], renamed["max_tier"]) == ("cheap", "mid")
    assert (unpinned["pinned_model"], unpinned["max_tier"]) == (None, "mid")
    assert (uncapped["pinned_model"], uncapped["max_tier"]) == (None, None)


def test_the_policy_shows_in_listings() -> None:
    client, _ = build()
    with client:
        headers = admin(client)
        client.post("/admin/tenants", json={"name": "acme", "max_tier": "slm"}, headers=headers)

        listed = {
            t["name"]: t for t in client.get("/admin/tenants", headers=headers).json()["data"]
        }

    assert listed["acme"]["max_tier"] == "slm"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"pinned_model": "nonexistent"}, "is not a configured model"),
        ({"pinned_model": "auto"}, "is not a configured model"),
        ({"max_tier": "gigantic"}, ""),
        ({"pinned_model": "top", "max_tier": "mid"}, "above the max_tier of mid"),
        ({"pinned_model": ""}, ""),
    ],
)
def test_impossible_policies_are_refused_when_set(body: dict[str, Any], message: str) -> None:
    client, _ = build()
    with client:
        response = client.post(
            "/admin/tenants", json={"name": "acme", **body}, headers=admin(client)
        )

    assert response.status_code == 400
    assert message in response.json()["error"]["message"]


def test_the_error_lists_the_models_that_do_exist() -> None:
    client, _ = build()
    with client:
        response = client.post(
            "/admin/tenants", json={"name": "a", "pinned_model": "nope"}, headers=admin(client)
        )

    assert "known: cheap, middle, top" in response.json()["error"]["message"]


def test_a_change_is_checked_against_what_is_already_set() -> None:
    client, _ = build()
    with client:
        headers = admin(client)
        pinned = client.post(
            "/admin/tenants", json={"name": "a", "pinned_model": "top"}, headers=headers
        ).json()["id"]
        capped = client.post(
            "/admin/tenants", json={"name": "b", "max_tier": "mid"}, headers=headers
        ).json()["id"]

        cap_below_pin = client.patch(
            f"/admin/tenants/{pinned}", json={"max_tier": "slm"}, headers=headers
        )
        pin_above_cap = client.patch(
            f"/admin/tenants/{capped}", json={"pinned_model": "top"}, headers=headers
        )

    assert cap_below_pin.status_code == pin_above_cap.status_code == 400


def test_a_cap_below_every_configured_model_is_refused_unless_a_model_is_pinned() -> None:
    client, _ = build(
        {
            "custom": (None, 1.0, 1.0),  # no tier, so a cap says nothing about it
            "middle": (Tier.MID, 0.6, 0.8),
            "top": (Tier.FRONTIER, 3.0, 15.0),
        }
    )
    with client:
        headers = admin(client)
        refused = client.post(
            "/admin/tenants", json={"name": "a", "max_tier": "slm"}, headers=headers
        )
        allowed = client.post(
            "/admin/tenants",
            json={"name": "b", "max_tier": "slm", "pinned_model": "custom"},
            headers=headers,
        )

    assert refused.status_code == 400
    assert "no configured model is at or below the tier slm" in refused.json()["error"]["message"]
    assert allowed.status_code == 201  # the pin means automatic routing is not needed


def test_an_empty_update_is_still_refused() -> None:
    client, _ = build()
    with client:
        headers = admin(client)
        tenant = client.post("/admin/tenants", json={"name": "a"}, headers=headers).json()["id"]

        assert client.patch(f"/admin/tenants/{tenant}", json={}, headers=headers).status_code == 400


# ---------- pinning a model ----------


def test_a_pinned_tenants_auto_requests_always_go_to_its_model() -> None:
    client, providers = build()
    with client:
        _, secret = organisation(client, "acme", pinned_model="top")

        replies = [answer(ask(client, secret, text)) for text in (EASY, MEDIUM, HARD)]

    assert replies == ["answer from top"] * 3
    assert providers["cheap"].calls == providers["middle"].calls == 0


def test_other_tenants_are_not_affected_by_a_pin() -> None:
    client, _ = build()
    with client:
        _, pinned = organisation(client, "pinned", pinned_model="top")
        _, plain = organisation(client, "plain")

        assert answer(ask(client, pinned, EASY)) == "answer from top"
        assert answer(ask(client, plain, EASY)) == "answer from cheap"


def test_the_response_says_the_model_was_pinned() -> None:
    client, _ = build()
    with client:
        _, secret = organisation(client, "acme", pinned_model="middle")

        headers = policy_headers(ask(client, secret, EASY))

    assert headers["x-llmrouter-model"] == "middle"
    assert headers["x-llmrouter-policy"] == "pinned"
    assert headers["x-llmrouter-reason"] == "pinned"
    assert headers["x-llmrouter-escalated"] == "false"
    assert "x-llmrouter-degraded" not in headers  # an easy request, a model more than strong enough


def test_a_pinned_model_weaker_than_the_request_needs_is_flagged() -> None:
    client, _ = build()
    with client:
        _, secret = organisation(client, "acme", pinned_model="cheap")

        response = ask(client, secret, HARD)

    assert answer(response) == "answer from cheap"  # the pin is honoured ...
    assert policy_headers(response)["x-llmrouter-degraded"] == "true"  # ... and the gap is visible


def test_a_pinned_request_still_reports_how_complex_it_was() -> None:
    client, _ = build()
    with client:
        _, secret = organisation(client, "acme", pinned_model="cheap")

        easy = policy_headers(ask(client, secret, EASY))
        hard = policy_headers(ask(client, secret, HARD))

    assert float(hard["x-llmrouter-complexity"]) > float(easy["x-llmrouter-complexity"])
    assert hard["x-llmrouter-required-tier"] == "frontier"


def test_a_pinned_model_is_not_escalated_but_does_use_its_configured_fallbacks() -> None:
    client, providers = build(fail={"cheap": 99})
    with client:
        _, secret = organisation(client, "acme", pinned_model="cheap")

        down = ask(client, secret, EASY)

    assert down.status_code == 502  # no escalation: the tenant asked for exactly this model
    assert providers["middle"].calls == providers["top"].calls == 0

    client, providers = build(fail={"cheap": 99}, fallbacks={"cheap": ["middle"]})
    with client:
        _, secret = organisation(client, "acme", pinned_model="cheap")

        rescued = ask(client, secret, EASY)

    assert answer(rescued) == "answer from middle"  # operator-configured fallbacks still apply


def test_a_model_without_a_tier_can_be_pinned() -> None:
    client, _ = build({"custom": (None, 1.0, 1.0), "cheap": (Tier.SLM, 0.05, 0.08)})
    with client:
        _, secret = organisation(client, "acme", pinned_model="custom")

        response = ask(client, secret, EASY)

    assert answer(response) == "answer from custom"
    assert policy_headers(response)["x-llmrouter-tier"] == "none"


def test_auto_is_available_to_a_pinned_tenant_even_with_no_tiered_models() -> None:
    client, _ = build({"only": (None, 1.0, 1.0)})
    with client:
        _, pinned = organisation(client, "pinned", pinned_model="only")
        _, plain = organisation(client, "plain")

        pinned_models = [
            m["id"] for m in client.get("/v1/models", headers=client.as_key(pinned)).json()["data"]
        ]
        plain_models = [
            m["id"] for m in client.get("/v1/models", headers=client.as_key(plain)).json()["data"]
        ]
        plain_auto = ask(client, plain, EASY)

    assert pinned_models == ["auto", "only"]
    assert plain_models == ["only"]
    assert plain_auto.status_code == 404


def test_quotas_price_the_pinned_model() -> None:
    client, providers = build()
    with client:
        headers = admin(client)
        tenant = client.post(
            "/admin/tenants", json={"name": "acme", "pinned_model": "top"}, headers=headers
        ).json()["id"]
        secret = client.post(
            f"/admin/tenants/{tenant}/keys",
            json={"name": "k", "monthly_budget_usd": 0.00001},  # fine for cheap, not for top
            headers=headers,
        ).json()["secret"]

        response = ask(client, secret, EASY, max_tokens=50)  # an easy request, but pinned to top

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "budget_exceeded"
    assert providers["top"].calls == 0


def test_a_key_still_needs_auto_in_its_allow_list_even_when_pinned() -> None:
    client, _ = build()
    with client:
        headers = admin(client)
        tenant = client.post(
            "/admin/tenants", json={"name": "acme", "pinned_model": "top"}, headers=headers
        ).json()["id"]
        secret = client.post(
            f"/admin/tenants/{tenant}/keys",
            json={"name": "k", "allowed_models": ["cheap"]},
            headers=headers,
        ).json()["secret"]

        response = ask(client, secret, EASY)

    assert response.status_code == 403 and response.json()["error"]["code"] == "model_not_allowed"


# ---------- capping the tier ----------


def test_a_capped_tenant_never_gets_a_stronger_model_than_its_cap() -> None:
    client, providers = build()
    with client:
        _, secret = organisation(client, "acme", max_tier="mid")

        hard = ask(client, secret, HARD)
        easy = ask(client, secret, EASY)

    assert answer(hard) == "answer from middle"  # needed frontier, capped at mid
    assert answer(easy) == "answer from cheap"  # ordinary routing below the cap
    assert providers["top"].calls == 0


def test_a_request_the_cap_prevents_serving_properly_is_flagged() -> None:
    client, _ = build()
    with client:
        _, secret = organisation(client, "acme", max_tier="mid")

        capped = policy_headers(ask(client, secret, HARD))
        within = policy_headers(ask(client, secret, EASY))

    assert capped["x-llmrouter-policy"] == "max-tier:mid"
    assert capped["x-llmrouter-required-tier"] == "frontier"
    assert capped["x-llmrouter-degraded"] == "true"
    assert within["x-llmrouter-policy"] == "max-tier:mid"
    assert "x-llmrouter-degraded" not in within


def test_the_tightest_cap_keeps_everything_on_the_small_model() -> None:
    client, providers = build()
    with client:
        _, secret = organisation(client, "acme", max_tier="slm")

        replies = [answer(ask(client, secret, text)) for text in (EASY, MEDIUM, HARD)]

    assert replies == ["answer from cheap"] * 3
    assert providers["middle"].calls == providers["top"].calls == 0


def test_escalation_stops_at_the_cap() -> None:
    client, providers = build(fail={"cheap": 99, "middle": 99})
    with client:
        _, secret = organisation(client, "acme", max_tier="mid")
        _, uncapped = organisation(client, "free")

        capped = ask(client, secret, EASY)
        free = ask(client, uncapped, EASY)

    assert capped.status_code == 502  # top is off limits, so nothing was left to try
    assert answer(free) == "answer from top"  # the same failures, no cap: escalated all the way
    assert providers["top"].calls == 1  # only the uncapped tenant ever reached it


def test_a_cap_does_not_restrict_requests_that_name_a_model() -> None:
    client, _ = build()
    with client:
        _, secret = organisation(client, "acme", max_tier="slm", pinned_model=None)

        direct = ask(client, secret, EASY, model="top")

    assert answer(direct) == "answer from top"  # the cap is about automatic routing only
    assert policy_headers(direct) == {}


def test_a_pin_does_not_restrict_requests_that_name_a_model() -> None:
    client, _ = build()
    with client:
        _, secret = organisation(client, "acme", pinned_model="top")

        direct = ask(client, secret, EASY, model="cheap")

    assert answer(direct) == "answer from cheap"


# ---------- changing a policy ----------


def test_a_change_applies_to_the_very_next_request() -> None:
    client, _ = build()
    with client:
        headers = admin(client)
        tenant, secret = organisation(client, "acme")
        assert answer(ask(client, secret, EASY)) == "answer from cheap"

        client.patch(f"/admin/tenants/{tenant}", json={"pinned_model": "top"}, headers=headers)
        pinned = answer(ask(client, secret, EASY))
        client.patch(f"/admin/tenants/{tenant}", json={"pinned_model": None}, headers=headers)
        released = answer(ask(client, secret, EASY))

    assert (pinned, released) == ("answer from top", "answer from cheap")


# ---------- a policy that has gone stale ----------


def corrupt(client: AuthedClient, tenant_id: str, **columns: Any) -> None:
    """Write a policy straight into the database, as if the config had changed underneath it."""
    app: Any = client.app

    async def apply() -> None:
        async with app.state.db_session_factory() as session:
            await session.execute(update(Tenant).where(Tenant.id == tenant_id).values(**columns))
            await session.commit()

    assert client.portal is not None
    client.portal.call(apply)


def test_a_pin_to_a_model_that_no_longer_exists_fails_clearly_not_silently() -> None:
    client, providers = build()
    with client:
        tenant, secret = organisation(client, "acme", pinned_model="top")
        corrupt(client, tenant, routing_pinned_model="retired-model")

        auto = ask(client, secret, EASY)
        direct = ask(client, secret, EASY, model="cheap")

    assert auto.status_code == 503
    assert auto.json()["error"]["code"] == "tenant_policy_invalid"
    assert "retired-model" not in auto.text  # no internal detail for the caller
    assert providers["cheap"].calls == 1  # only the direct request got through
    assert direct.status_code == 200


def test_a_stale_pin_never_falls_back_to_ordinary_routing() -> None:
    """A pin may be a compliance rule, so quietly routing elsewhere would be the worst outcome."""
    client, providers = build()
    with client:
        tenant, secret = organisation(client, "acme", pinned_model="top")
        corrupt(client, tenant, routing_pinned_model="retired-model")

        ask(client, secret, EASY)

    assert sum(p.calls for p in providers.values()) == 0


def test_a_pin_and_a_cap_that_now_conflict_fail_clearly() -> None:
    client, _ = build()
    with client:
        tenant, secret = organisation(client, "acme", pinned_model="top")
        corrupt(client, tenant, routing_max_tier="slm")

        response = ask(client, secret, EASY)

    assert (
        response.status_code == 503 and response.json()["error"]["code"] == "tenant_policy_invalid"
    )


def test_a_cap_that_leaves_no_model_fails_clearly() -> None:
    client, _ = build({"middle": (Tier.MID, 0.6, 0.8), "top": (Tier.FRONTIER, 3.0, 15.0)})
    with client:
        tenant, secret = organisation(client, "acme")
        corrupt(client, tenant, routing_max_tier="slm")

        response = ask(client, secret, EASY)

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "routing_policy_no_model"


def test_a_stale_policy_is_logged_for_the_operator(caplog: pytest.LogCaptureFixture) -> None:
    client, _ = build()
    with client:
        tenant, secret = organisation(client, "acme", pinned_model="top")
        corrupt(client, tenant, routing_pinned_model="retired-model")

        with caplog.at_level(logging.ERROR):
            ask(client, secret, EASY)

    assert any("retired-model" in r.getMessage() for r in caplog.records)  # the operator sees it


# ---------- isolation between organisations ----------


def test_concurrent_requests_follow_their_own_organisations_policy() -> None:
    client, _ = build()
    with client:
        _, plain = organisation(client, "plain")
        _, pinned = organisation(client, "pinned", pinned_model="middle")
        _, capped = organisation(client, "capped", max_tier="slm")
        callers = [
            (plain, HARD, "answer from top"),
            (pinned, HARD, "answer from middle"),
            (capped, HARD, "answer from cheap"),
        ]

        def call(index: int) -> tuple[str, str]:
            secret, text, expected = callers[index % 3]
            return expected, answer(ask(client, secret, text))

        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(call, range(45)))

    assert all(got == expected for expected, got in results)


# ---------- the decision log ----------


def test_the_log_records_the_policy(caplog: pytest.LogCaptureFixture) -> None:
    client, _ = build()
    with client:
        _, pinned = organisation(client, "pinned", pinned_model="middle")
        _, capped = organisation(client, "capped", max_tier="mid")
        _, plain = organisation(client, "plain")

        with caplog.at_level(logging.INFO, logger="app.routing.decisions"):
            ask(client, pinned, EASY)
            ask(client, capped, EASY)
            ask(client, plain, EASY)

    decisions = [r for r in caplog.records if r.getMessage() == "routing decision"]
    assert [(d.pinned, d.policy) for d in decisions] == [  # type: ignore[attr-defined]
        (True, "pinned"),
        (False, "max-tier:mid"),
        (False, None),
    ]


# ---------- the routing functions on their own ----------


def raw(content: str = EASY) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {"model": "auto", "messages": [{"role": "user", "content": content}]}
    )


def ladder() -> ModelRegistry:
    return ModelRegistry(
        [
            Deployment.create(
                model_config(name=n, model=f"{n}-up", tier=t, input_cost_per_mtok=p),
                FakeProvider(n),
            )
            for n, (t, p) in {
                "cheap": (Tier.SLM, 0.1),
                "middle": (Tier.MID, 1.0),
                "top": (Tier.FRONTIER, 5.0),
            }.items()
        ]
    )


def test_the_cap_removes_stronger_models_from_the_ranking() -> None:
    choice = choose_model(raw(HARD), ladder(), LatencyTracker(), max_tier=Tier.MID)

    assert [c.alias for c in choice.candidates] == ["middle"]
    assert choice.policy == "max-tier:mid" and not choice.clears_bar
    assert choice.escalation == []


def test_a_cap_that_removes_nothing_still_labels_the_choice() -> None:
    choice = choose_model(raw(EASY), ladder(), LatencyTracker(), max_tier=Tier.FRONTIER)

    assert choice.alias == "cheap" and choice.policy == "max-tier:frontier" and choice.clears_bar


def test_without_a_cap_the_choice_carries_no_policy() -> None:
    assert choose_model(raw(), ladder(), LatencyTracker()).policy is None


def test_a_cap_below_every_model_is_an_error() -> None:
    registry = ModelRegistry(
        [Deployment.create(model_config(name="m", model="m", tier=Tier.MID), FakeProvider())]
    )

    with pytest.raises(NoRoutableModelError) as excinfo:
        choose_model(raw(), registry, LatencyTracker(), max_tier=Tier.SLM)

    assert excinfo.value.kind == "policy"


def test_a_pinned_choice_names_exactly_the_pinned_model() -> None:
    choice = pinned_choice(raw(HARD), ladder(), LatencyTracker(), "cheap")

    assert choice.alias == "cheap" and choice.pinned and choice.policy == "pinned"
    assert len(choice.candidates) == 1 and choice.escalation == []
    assert not choice.clears_bar  # too weak for a hard request


def test_pinning_an_unknown_model_is_an_error() -> None:
    with pytest.raises(UnknownModelError):
        pinned_choice(raw(), ladder(), LatencyTracker(), "ghost")


def test_a_request_too_big_for_the_pinned_model_is_a_context_error() -> None:
    registry = ModelRegistry(
        [
            Deployment.create(
                model_config(name="m", model="m", tier=Tier.SLM, context_window=100), FakeProvider()
            )
        ]
    )

    with pytest.raises(NoRoutableModelError) as excinfo:
        pinned_choice(raw("word " * 2_000), registry, LatencyTracker(), "m")

    assert excinfo.value.kind == "context" and excinfo.value.window == 100


def test_resolving_with_a_policy() -> None:
    registry = ladder()

    pinned, pinned_choice_ = resolve_model(raw(), registry, LatencyTracker(), pinned_model="top")
    capped, capped_choice = resolve_model(raw(HARD), registry, LatencyTracker(), max_tier=Tier.SLM)
    plain, plain_choice = resolve_model(raw(), registry, LatencyTracker())

    assert (pinned.model, capped.model, plain.model) == ("top", "cheap", "cheap")
    assert pinned_choice_ is not None and pinned_choice_.pinned
    assert capped_choice is not None and capped_choice.policy == "max-tier:slm"
    assert plain_choice is not None and plain_choice.policy is None


def test_resolving_reports_unusable_policies_as_such() -> None:
    registry = ladder()

    with pytest.raises(TenantPolicyError) as unknown:
        resolve_model(raw(), registry, LatencyTracker(), pinned_model="ghost")
    with pytest.raises(TenantPolicyError) as conflict:
        resolve_model(raw(), registry, LatencyTracker(), pinned_model="top", max_tier=Tier.MID)

    assert unknown.value.kind == conflict.value.kind == "pin_unavailable"
    assert "above the cap" in str(conflict.value)


def test_a_policy_is_ignored_for_requests_that_do_not_use_auto() -> None:
    request = ChatCompletionRequest.model_validate(
        {"model": "top", "messages": [{"role": "user", "content": "hi"}]}
    )

    resolved, choice = resolve_model(request, ladder(), LatencyTracker(), pinned_model="cheap")

    assert resolved is request and choice is None


# ---------- storage ----------


async def test_the_tenant_service_stores_and_updates_the_policy() -> None:
    from app.db.base import Base
    from app.db.session import create_engine, create_session_factory

    engine = create_engine("sqlite+aiosqlite://")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        async with create_session_factory(engine)() as session:
            tenant = await create_tenant(session, "acme", pinned_model=" top ", max_tier=Tier.MID)
            assert (tenant.routing_pinned_model, tenant.routing_max_tier) == ("top", "mid")

            kept = await update_tenant(session, tenant.id, name="acme2")
            assert (kept.routing_pinned_model, kept.routing_max_tier) == ("top", "mid")

            cleared = await update_tenant(session, tenant.id, pinned_model=None)
            assert (cleared.routing_pinned_model, cleared.routing_max_tier) == (None, "mid")

            with pytest.raises(ValueError, match="pinned_model must be"):
                await update_tenant(session, tenant.id, pinned_model="two words")
    finally:
        await engine.dispose()


def test_the_policy_migration_keeps_existing_tenants(tmp_path: Path) -> None:
    database = tmp_path / "policy.db"
    config = alembic_config(database)
    command.upgrade(config, "0007")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO tenants (id, name, is_active, created_at)"
            " VALUES ('t1', 'acme', 1, '2026-01-01')"
        )

    command.upgrade(config, "head")

    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT name, routing_pinned_model, routing_max_tier FROM tenants"
        ).fetchone()
    assert row == ("acme", None, None)


def test_the_policy_migration_can_be_reversed(tmp_path: Path) -> None:
    database = tmp_path / "policy.db"
    config = alembic_config(database)
    command.upgrade(config, "head")

    command.downgrade(config, "0007")

    with sqlite3.connect(database) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(tenants)")]
    assert "routing_pinned_model" not in columns and "routing_max_tier" not in columns
    assert "monthly_budget_nano_usd" in columns
