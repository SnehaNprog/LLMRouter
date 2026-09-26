import random
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
import pytest

from app.core.config import Settings
from app.core.tiers import Tier
from app.main import create_app
from app.providers.errors import BadRequest, ProviderDown, Timeout
from app.providers.fake import FakeProvider
from app.quota.budget import BudgetStore, usd_to_nano
from app.quota.budget_limiter import BudgetLimiter
from app.quota.reservation import NO_RESERVATION, Reservation
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry, UnknownModelError
from app.router.retry import RetryPolicy
from app.router.router import Router
from app.routing.score import ComplexityScore
from app.routing.selector import Candidate, RoutingChoice
from app.schemas.chat import ChatCompletionRequest, Message
from app.schemas.completion import ChatCompletionChunk
from tests.conftest import AuthedClient, model_config
from tests.test_auto_routing import EASY, HARD, MEDIUM, answer, ask


class FirstChoice(random.Random):
    def choices(  # type: ignore[override]
        self, population: Sequence[Any], weights: Any = None, **kwargs: Any
    ) -> list[Any]:
        return [population[0]]

    def uniform(self, a: float, b: float) -> float:
        return b


async def no_sleep(_: float) -> None:
    return None


def dep(name: str, provider: FakeProvider, tier: Tier = Tier.SLM, **config: Any) -> Deployment:
    return Deployment.create(
        model_config(name=name, model=f"{name}-up", tier=tier, **config), provider
    )


def request_for(alias: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(model=alias, messages=[Message(role="user", content="hi")])


def setup(
    attempts: int = 2, fallbacks: dict[str, list[str]] | None = None, **fails: int
) -> tuple[Router, dict[str, FakeProvider]]:
    providers = {
        name: FakeProvider(name, reply=f"answer from {name}", fail_first=fails.get(name, 0))
        for name in ("cheap", "middle", "top", "spare")
    }
    registry = ModelRegistry(
        [
            dep(name, provider, tier)
            for (name, provider), tier in zip(
                providers.items(), (Tier.SLM, Tier.MID, Tier.FRONTIER, Tier.FRONTIER), strict=True
            )
        ],
        fallbacks,
    )
    router = Router(
        registry,
        policy=RetryPolicy(max_attempts=attempts, base_delay=0.0),
        rng=FirstChoice(),
        sleep=no_sleep,
    )
    return router, providers


async def reply_of(router: Router, alias: str, **kwargs: Any) -> str:
    response = await router.complete(request_for(alias), **kwargs)
    return str(response.choices[0].message.content)


# ---------- the router walks the escalation path ----------


async def test_a_healthy_model_is_never_escalated_from() -> None:
    router, providers = setup()

    assert await reply_of(router, "cheap", escalation=["middle", "top"]) == "answer from cheap"
    assert (providers["middle"].calls, providers["top"].calls) == (0, 0)


async def test_when_the_chosen_model_fails_the_next_one_answers() -> None:
    router, providers = setup(cheap=99)

    assert await reply_of(router, "cheap", escalation=["middle", "top"]) == "answer from middle"
    assert providers["top"].calls == 0


async def test_escalation_continues_up_the_path_until_something_answers() -> None:
    router, providers = setup(cheap=99, middle=99)

    assert await reply_of(router, "cheap", escalation=["middle", "top"]) == "answer from top"
    assert providers["cheap"].calls == providers["middle"].calls == 2  # each was retried first


async def test_when_the_whole_path_fails_the_last_error_is_raised() -> None:
    router, providers = setup(cheap=99, middle=99, top=99)
    providers["top"].fail_first = 99

    with pytest.raises(ProviderDown):
        await reply_of(router, "cheap", escalation=["middle", "top"])


async def test_the_chosen_model_is_retried_before_anything_is_escalated() -> None:
    router, providers = setup(attempts=3, cheap=2)  # fails twice, then works

    assert await reply_of(router, "cheap", escalation=["middle"]) == "answer from cheap"
    assert providers["cheap"].calls == 3 and providers["middle"].calls == 0


async def test_configured_fallbacks_come_before_escalation() -> None:
    router, providers = setup(fallbacks={"cheap": ["spare"]}, cheap=99)

    assert await reply_of(router, "cheap", escalation=["middle"]) == "answer from spare"
    assert providers["middle"].calls == 0


async def test_a_model_already_tried_as_a_fallback_is_not_tried_again() -> None:
    router, providers = setup(fallbacks={"cheap": ["middle"]}, cheap=99, middle=99)

    assert await reply_of(router, "cheap", escalation=["middle", "top"]) == "answer from top"
    assert providers["middle"].calls == 2  # one visit (two attempts), not two visits


async def test_escalation_targets_bring_their_own_fallbacks() -> None:
    router, _ = setup(fallbacks={"middle": ["spare"]}, cheap=99, middle=99)

    assert await reply_of(router, "cheap", escalation=["middle", "top"]) == "answer from spare"


async def test_a_bad_request_is_never_escalated() -> None:
    router, providers = setup()
    providers["cheap"].fail_first = 99
    providers["cheap"]._error_factory = lambda: BadRequest("the request itself is wrong")

    with pytest.raises(BadRequest):
        await reply_of(router, "cheap", escalation=["middle", "top"])

    assert providers["middle"].calls == 0


async def test_without_an_escalation_path_nothing_changes() -> None:
    router, providers = setup(cheap=99)

    with pytest.raises(ProviderDown):
        await reply_of(router, "cheap")

    assert providers["middle"].calls == 0


async def test_an_unknown_escalation_target_is_reported() -> None:
    router, _ = setup(cheap=99)

    with pytest.raises(UnknownModelError):
        await reply_of(router, "cheap", escalation=["nonexistent"])


async def test_the_answering_model_is_reported() -> None:
    from app.router.router import Served

    router, providers = setup(cheap=99)
    served = Served()

    await reply_of(router, "cheap", escalation=["middle"], served=served)

    assert served.deployment is not None and served.deployment.alias == "middle"


async def test_each_escalation_is_judged_by_its_own_failure() -> None:
    router, providers = setup()
    providers["cheap"].fail_first = 99
    providers["cheap"]._error_factory = lambda: Timeout("slow")
    providers["middle"].fail_first = 99

    assert await reply_of(router, "cheap", escalation=["middle", "top"]) == "answer from top"


# ---------- streaming ----------


async def collect(stream: AsyncIterator[ChatCompletionChunk]) -> str:
    return "".join([c.choices[0].delta.content or "" async for c in stream if c.choices])


async def test_a_stream_that_cannot_start_escalates() -> None:
    router, providers = setup(cheap=99)

    text = await collect(router.stream(request_for("cheap"), escalation=["middle"]))

    assert text == "answer from middle"
    assert providers["middle"].calls == 1


async def test_a_stream_that_has_begun_is_not_escalated() -> None:
    class DiesMidStream(FakeProvider):
        async def stream(self, request: Any) -> AsyncIterator[ChatCompletionChunk]:
            self.calls += 1
            async for chunk in FakeProvider(reply="a b c").stream(request):
                yield chunk
                raise ProviderDown("reset", provider="p")

    router, providers = setup()
    registry = ModelRegistry(
        [dep("cheap", DiesMidStream("cheap")), dep("middle", providers["middle"], Tier.MID)]
    )
    router = Router(registry, policy=RetryPolicy(max_attempts=1), sleep=no_sleep)

    with pytest.raises(ProviderDown):
        await collect(router.stream(request_for("cheap"), escalation=["middle"]))

    assert providers["middle"].calls == 0


# ---------- admission: the path can be restricted ----------


async def test_the_gate_is_asked_about_escalation_targets_only() -> None:
    router, _ = setup(cheap=99)
    asked: list[str] = []

    async def gate(alias: str) -> bool:
        asked.append(alias)
        return True

    await reply_of(router, "cheap", escalation=["middle", "top"], admit=gate)

    assert asked == ["middle"]  # never for the chosen model, and not past the one that answered


async def test_a_refused_target_is_skipped_for_the_next() -> None:
    router, providers = setup(cheap=99)

    async def gate(alias: str) -> bool:
        return alias != "middle"

    assert (
        await reply_of(router, "cheap", escalation=["middle", "top"], admit=gate)
        == "answer from top"
    )
    assert providers["middle"].calls == 0


async def test_when_every_target_is_refused_the_original_failure_is_raised() -> None:
    router, providers = setup(cheap=99)

    async def gate(alias: str) -> bool:
        return False

    with pytest.raises(ProviderDown):
        await reply_of(router, "cheap", escalation=["middle", "top"], admit=gate)

    assert providers["middle"].calls == providers["top"].calls == 0


async def test_configured_fallbacks_are_not_put_through_the_gate() -> None:
    router, _ = setup(fallbacks={"cheap": ["spare"]}, cheap=99)
    asked: list[str] = []

    async def gate(alias: str) -> bool:
        asked.append(alias)
        return True

    await reply_of(router, "cheap", escalation=["middle"], admit=gate)

    assert asked == []  # the spare answered; the gate only guards the extra escalation


# ---------- the order of the escalation path ----------


def candidate(alias: str, tier: Tier, utility: float) -> Candidate:
    return Candidate(alias, tier, 0.0, 0.0, True, utility)


def choice_of(*candidates: Candidate) -> RoutingChoice:
    return RoutingChoice(
        alias=candidates[0].alias,
        complexity=ComplexityScore(0.1, {}, False),
        required_tier=Tier.SLM,
        clears_bar=True,
        candidates=list(candidates),
    )


def test_escalation_lists_the_other_candidates_in_ranked_order() -> None:
    choice = choice_of(
        candidate("a", Tier.SLM, 0.1),
        candidate("b", Tier.MID, 0.4),
        candidate("c", Tier.FRONTIER, 0.9),
    )

    assert choice.escalation == ["b", "c"]


def test_stronger_or_equal_models_come_before_weaker_ones() -> None:
    choice = choice_of(
        candidate("chosen", Tier.MID, 0.1),
        candidate("weaker", Tier.SLM, 0.2),  # ranks second, but is a step down
        candidate("sibling", Tier.MID, 0.3),
        candidate("stronger", Tier.FRONTIER, 0.5),
    )

    assert choice.escalation == ["sibling", "stronger", "weaker"]


def test_a_lone_candidate_has_nowhere_to_escalate() -> None:
    assert choice_of(candidate("only", Tier.SLM, 0.0)).escalation == []


# ---------- paying for a dearer model ----------

PRICED = {"input_cost_per_mtok": 100.0, "output_cost_per_mtok": 200.0}


async def test_a_reservation_can_be_raised_and_the_extra_is_charged() -> None:
    charged: list[int] = []

    async def grow(extra: int) -> bool:
        charged.append(extra)
        return True

    reservation = Reservation(100, 0, None, grow=grow)

    assert await reservation.ensure(350) is True
    assert (charged, reservation.reserved) == ([250], 350)
    assert await reservation.ensure(300) is True  # already covered: no further charge
    assert charged == [250]


async def test_a_reservation_that_cannot_be_raised_stays_as_it_was() -> None:
    async def refuse(extra: int) -> bool:
        return False

    reservation = Reservation(100, 0, None, grow=refuse)

    assert await reservation.ensure(900) is False
    assert reservation.reserved == 100


async def test_a_reservation_with_nothing_to_charge_always_admits() -> None:
    assert await NO_RESERVATION.ensure(10**9) is True


def budget_setup() -> tuple[BudgetLimiter, BudgetStore, ModelRegistry]:
    import fakeredis

    store = BudgetStore(fakeredis.FakeAsyncRedis(decode_responses=True))
    registry = ModelRegistry(
        [
            dep(
                "cheap", FakeProvider(), Tier.SLM, input_cost_per_mtok=1.0, output_cost_per_mtok=1.0
            ),
            dep("top", FakeProvider(), Tier.FRONTIER, **PRICED),
        ]
    )
    return BudgetLimiter(store), store, registry


def auth_with_budget(usd: float) -> Any:
    from tests.test_budget_limit import auth

    return auth(key_budget=usd)


async def test_escalating_to_a_dearer_model_charges_the_difference() -> None:
    limiter, store, registry = budget_setup()
    request = request_for("cheap").model_copy(update={"max_tokens": 10})
    reservation = await limiter.reserve(auth_with_budget(1.0), request, registry, 8)
    cheap_price = reservation.reserved
    top_price = limiter.estimate_for(registry, request, 8, ["top"])

    assert await reservation.ensure(top_price) is True

    assert top_price > cheap_price
    assert await store.spent("key:key-1:2026-11") == top_price  # exactly the worst case, no more


async def test_a_budget_that_cannot_cover_the_dearer_model_refuses_and_changes_nothing() -> None:
    limiter, store, registry = budget_setup()
    request = request_for("cheap").model_copy(update={"max_tokens": 10})
    top_price = limiter.estimate_for(registry, request, 8, ["top"])
    reservation = await limiter.reserve(auth_with_budget(top_price / 2 / 1e9), request, registry, 8)
    before = await store.spent("key:key-1:2026-11")

    assert await reservation.ensure(top_price) is False

    assert await store.spent("key:key-1:2026-11") == before
    assert reservation.reserved < top_price


async def test_settling_after_a_raise_refunds_against_the_raised_amount() -> None:
    limiter, store, registry = budget_setup()
    request = request_for("cheap").model_copy(update={"max_tokens": 10})
    reservation = await limiter.reserve(auth_with_budget(1.0), request, registry, 8)
    top_price = limiter.estimate_for(registry, request, 8, ["top"])
    await reservation.ensure(top_price)

    await reservation.settle(1_234)  # what the dearer model really cost

    assert await store.spent("key:key-1:2026-11") == 1_234


async def test_a_budget_cannot_be_dodged_by_starting_on_a_free_model() -> None:
    import fakeredis

    store = BudgetStore(fakeredis.FakeAsyncRedis(decode_responses=True))
    registry = ModelRegistry(
        [dep("local", FakeProvider()), dep("top", FakeProvider(), Tier.FRONTIER, **PRICED)]
    )
    limiter = BudgetLimiter(store)
    request = request_for("local").model_copy(update={"max_tokens": 10})
    reservation = await limiter.reserve(auth_with_budget(1e-9), request, registry, 8)  # ~nothing
    top_price = limiter.estimate_for(registry, request, 8, ["top"])

    assert reservation.reserved == 0
    assert await reservation.ensure(top_price) is False  # the free start did not skip the check


# ---------- the whole thing over HTTP ----------


def tiered(
    *,
    fail: dict[str, int] | None = None,
    fallbacks: dict[str, list[str]] | None = None,
    **settings: Any,
) -> tuple[AuthedClient, dict[str, FakeProvider]]:
    spec = {
        "cheap": (Tier.SLM, 0.05, 0.08),
        "middle": (Tier.MID, 0.6, 0.8),
        "top": (Tier.FRONTIER, 3.0, 15.0),
    }
    providers = {
        name: FakeProvider(name, reply=f"answer from {name}", fail_first=(fail or {}).get(name, 0))
        for name in spec
    }

    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return ModelRegistry(
            [
                dep(name, providers[name], tier, input_cost_per_mtok=a, output_cost_per_mtok=b)
                for name, (tier, a, b) in spec.items()
            ],
            fallbacks,
        )

    options = {"retry_base_delay": 0.0, "retry_max_attempts": 1, **settings}
    return AuthedClient(create_app(Settings(_env_file=None, **options), factory)), providers


def test_an_easy_request_whose_small_model_is_down_is_answered_by_a_stronger_one() -> None:
    client, providers = tiered(fail={"cheap": 99})
    with client:
        response = ask(client, None, EASY)

    assert response.status_code == 200
    assert answer(response) == "answer from middle"
    assert providers["cheap"].calls == 1


def test_it_keeps_escalating_until_something_answers() -> None:
    client, _ = tiered(fail={"cheap": 99, "middle": 99})
    with client:
        response = ask(client, None, EASY)

    assert answer(response) == "answer from top"


def test_when_everything_is_down_the_caller_gets_a_clean_502() -> None:
    client, _ = tiered(fail={"cheap": 99, "middle": 99, "top": 99})
    with client:
        response = ask(client, None, EASY)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_unavailable"


def test_a_hard_request_has_nowhere_stronger_to_go() -> None:
    client, providers = tiered(fail={"top": 99})
    with client:
        response = ask(client, None, HARD)  # needs the frontier tier, and only top is that

    assert response.status_code == 502
    assert providers["middle"].calls == providers["cheap"].calls == 0  # weaker ones were not enough


def test_a_medium_request_escalates_upward_before_anything_else() -> None:
    client, providers = tiered(fail={"middle": 99})
    with client:
        response = ask(client, None, MEDIUM)

    assert answer(response) == "answer from top"
    assert providers["cheap"].calls == 0  # candidates only ever include models good enough


def test_naming_a_model_directly_never_escalates() -> None:
    client, providers = tiered(fail={"cheap": 99})
    with client:
        response = ask(client, None, EASY, model="cheap")

    assert response.status_code == 502
    assert providers["middle"].calls == 0


def test_streaming_requests_escalate_before_the_first_chunk() -> None:
    client, providers = tiered(fail={"cheap": 99})
    with client:
        response = ask(client, None, EASY, stream=True)

    assert response.status_code == 200
    assert providers["middle"].calls == 1 and "[DONE]" in response.text


def test_escalation_counts_as_one_request_for_rate_limits() -> None:
    client, _ = tiered(fail={"cheap": 99})
    with client:
        secret = client.issue_key(rpm_limit=2)

        statuses = [ask(client, secret, EASY).status_code for _ in range(3)]

    assert statuses == [200, 200, 429]  # each escalated request used one, not two


def test_the_answering_models_price_is_what_is_charged() -> None:
    client, _ = tiered(fail={"cheap": 99})
    with client:
        secret = client.issue_key(monthly_budget_usd=1.0)

        ask(client, secret, EASY, max_tokens=50)

        app: Any = client.app
        assert client.portal is not None

        async def spent() -> list[int]:
            return [
                int(await app.state.redis.get(k))
                async for k in app.state.redis.scan_iter("*budget:key:*")
            ]

        charged = client.portal.call(spent)

    # middle answered: 10 prompt + 5 completion tokens at $0.60 / $0.80 per million
    assert charged == [usd_to_nano((10 * 0.6 + 5 * 0.8) / 1_000_000)]


def test_a_budget_that_cannot_afford_the_stronger_model_stops_the_escalation() -> None:
    client, providers = tiered(fail={"cheap": 99, "middle": 99})
    with client:
        # Room for the cheap and middle worst cases (~$0.0001) but not the frontier one ($0.0008).
        secret = client.issue_key(monthly_budget_usd=0.0002)

        response = ask(client, secret, EASY, max_tokens=50)

        app: Any = client.app
        assert client.portal is not None

        async def spent() -> list[int]:
            return [
                int(await app.state.redis.get(k))
                async for k in app.state.redis.scan_iter("*budget:key:*")
            ]

        charged = client.portal.call(spent)

    assert response.status_code == 502  # the original failure, not a budget error
    assert providers["top"].calls == 0  # never contacted: the key could not have paid for it
    assert charged == [0]  # and the failed request cost nothing


def test_a_generous_budget_lets_the_escalation_reach_the_frontier_model() -> None:
    client, _ = tiered(fail={"cheap": 99, "middle": 99})
    with client:
        secret = client.issue_key(monthly_budget_usd=1.0)

        response = ask(client, secret, EASY, max_tokens=50)

    assert answer(response) == "answer from top"


def test_the_tenant_budget_also_gates_escalation() -> None:
    client, providers = tiered(fail={"cheap": 99, "middle": 99})
    with client:
        admin = client.as_key(client.issue_key(["admin:tenants"]))
        tenant = client.get("/admin/tenants", headers=admin).json()["data"][0]["id"]
        client.patch(f"/admin/tenants/{tenant}", json={"monthly_budget_usd": 0.0002}, headers=admin)

        response = ask(client, None, EASY, max_tokens=50)

    assert response.status_code == 502 and providers["top"].calls == 0
