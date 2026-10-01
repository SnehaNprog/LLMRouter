import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import select
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.chat import _event_stream
from app.api.errors import outcome_of
from app.auth.errors import InsufficientScopeError, InvalidApiKeyError
from app.core.config import Settings
from app.core.tiers import Tier
from app.db.models import RequestLog, Tenant
from app.main import create_app
from app.providers.errors import (
    AuthFailed,
    BadRequest,
    ProviderDown,
    RateLimited,
    Timeout,
)
from app.providers.fake import FakeProvider
from app.quota.budget_limiter import BudgetExceededError
from app.quota.guard import QuotaUnavailableError
from app.quota.headers import Window
from app.quota.limiter import RateLimitExceededError, TokenLimitExceededError
from app.router.context import ContextLengthExceededError
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry, UnknownModelError
from app.router.router import Served
from app.routing.resolve import TenantPolicyError
from app.usage.record import LogEntry, RequestLogger
from tests.conftest import TEST_ALIAS, AuthedClient, model_config, seed_credentials
from tests.test_tpm_limit import Recorder, chunks, text_chunk

PRICED = {"input_cost_per_mtok": 10.0, "output_cost_per_mtok": 30.0}  # per million tokens
BODY = {"model": TEST_ALIAS, "messages": [{"role": "user", "content": "hello there"}]}


def app_client(*deployments: tuple[FakeProvider, dict[str, Any]], **settings: Any) -> AuthedClient:
    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return ModelRegistry(
            Deployment.create(model_config(**config), provider) for provider, config in deployments
        )

    options = {"retry_base_delay": 0.0, "retry_max_attempts": 1, **settings}
    return AuthedClient(create_app(Settings(_env_file=None, **options), factory))


def single(provider: FakeProvider | None = None, **config: Any) -> AuthedClient:
    return app_client((provider or FakeProvider(), {"name": TEST_ALIAS, **PRICED, **config}))


def logged(client: AuthedClient) -> list[RequestLog]:
    async def fetch() -> list[RequestLog]:
        async with client.app.state.db_session_factory() as session:  # type: ignore[attr-defined]
            result = await session.execute(select(RequestLog).order_by(RequestLog.created_at))
            return list(result.scalars())

    assert client.portal is not None
    return client.portal.call(fetch)  # type: ignore[no-any-return]


def tenant_id(client: AuthedClient) -> str:
    async def fetch() -> str:
        async with client.app.state.db_session_factory() as session:  # type: ignore[attr-defined]
            return str((await session.execute(select(Tenant.id))).scalars().first())

    assert client.portal is not None
    return client.portal.call(fetch)  # type: ignore[no-any-return]


def chat(client: AuthedClient, **extra: Any) -> httpx.Response:
    return client.post("/v1/chat/completions", json={**BODY, **extra})


# ---------- what is logged for a good request ----------


def test_a_completion_is_logged_with_who_what_and_how_much() -> None:
    with single(FakeProvider(prompt_tokens=100, completion_tokens=50)) as client:
        response = chat(client)
        [row] = logged(client)
        expected_tenant = tenant_id(client)

    assert response.status_code == 200
    assert row.tenant_id == expected_tenant
    assert row.key_id
    assert (row.requested_model, row.served_model) == (TEST_ALIAS, TEST_ALIAS)
    assert row.deployment_id and row.deployment_id.startswith("dep_")
    assert (row.status_code, row.error_code, row.stream) == (200, None, False)
    assert (row.prompt_tokens, row.completion_tokens) == (100, 50)
    assert row.cost_nano_usd == (100 * 10 + 50 * 30) * 1000  # nano-dollars at $/Mtok rates
    assert row.latency_ms >= 0
    assert (row.routed, row.tier, row.escalated, row.complexity) == (False, None, False, None)
    assert row.request_id == response.headers["x-request-id"]
    assert row.day == row.created_at.strftime("%Y-%m-%d")


def test_one_row_per_request() -> None:
    with single() as client:
        for _ in range(4):
            chat(client)
        assert len(logged(client)) == 4


def test_a_streamed_completion_is_logged_once_it_ends_with_its_real_usage() -> None:
    with single(FakeProvider(stream_usage=True, prompt_tokens=100, completion_tokens=50)) as client:
        response = chat(client, stream=True)
        [row] = logged(client)

    assert response.status_code == 200
    assert row.stream is True and row.status_code == 200 and row.error_code is None
    assert (row.prompt_tokens, row.completion_tokens) == (100, 50)
    assert row.cost_nano_usd == (100 * 10 + 50 * 30) * 1000


def test_the_model_tier_is_recorded_when_the_model_has_one() -> None:
    with single(tier=Tier.MID) as client:
        chat(client)
        [row] = logged(client)
    assert row.tier == "mid" and row.routed is False


# ---------- what is never logged ----------


def test_neither_the_prompt_nor_the_reply_is_stored() -> None:
    secret_prompt = "my-very-private-question-7f3a"
    secret_reply = "an-equally-private-answer-91bc"
    with single(FakeProvider(reply=secret_reply)) as client:
        chat(client, messages=[{"role": "user", "content": secret_prompt}])
        [row] = logged(client)

    stored = " ".join(str(getattr(row, column.name)) for column in RequestLog.__table__.columns)
    assert secret_prompt not in stored and secret_reply not in stored


def test_the_log_has_no_column_that_could_hold_content() -> None:
    """Adding a column here is a decision about privacy; this test makes it a deliberate one."""
    assert {c.name for c in RequestLog.__table__.columns} == {
        "id", "created_at", "day", "request_id", "tenant_id", "key_id", "requested_model",
        "served_model", "deployment_id", "status_code", "error_code", "stream", "prompt_tokens",
        "completion_tokens", "cost_nano_usd", "latency_ms", "routed", "tier", "escalated",
        "complexity",
    }  # fmt: skip


# ---------- failures ----------


def test_a_failed_request_is_logged_as_the_client_saw_it() -> None:
    with single(FakeProvider(fail_first=99)) as client:
        response = chat(client)
        [row] = logged(client)

    assert response.status_code == 502
    assert (row.status_code, row.error_code) == (502, "upstream_unavailable")
    assert (row.prompt_tokens, row.completion_tokens, row.cost_nano_usd) == (0, 0, 0)
    assert row.served_model is None and row.requested_model == TEST_ALIAS


Scenario = Callable[[], tuple[AuthedClient, dict[str, Any], dict[str, Any]]]
SCENARIOS: dict[str, Scenario] = {
    "unknown model": lambda: (single(), {"model": "nope"}, {}),
    "model not allowed": lambda: (single(), {}, {"allowed_models": ["other"]}),
    "upstream down": lambda: (single(FakeProvider(fail_first=99)), {}, {}),
    "upstream timeout": lambda: (
        single(FakeProvider(fail_first=99, error_factory=lambda: Timeout("slow"))), {}, {}),
    "upstream rate limit": lambda: (
        single(FakeProvider(fail_first=99, error_factory=lambda: RateLimited("x", retry_after=5))),
        {}, {}),
    "upstream bad request": lambda: (
        single(FakeProvider(fail_first=99, error_factory=lambda: BadRequest("no"))), {}, {}),
    "upstream auth": lambda: (
        single(FakeProvider(fail_first=99, error_factory=lambda: AuthFailed("no"))), {}, {}),
    "budget": lambda: (single(), {}, {"monthly_budget_usd": 0.000001}),
    "request too large": lambda: (single(), {}, {"tpm_limit": 5}),
    "context length": lambda: (single(context_window=10), {"max_tokens": 100}, {}),
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_kind_of_failure_is_logged_with_the_status_and_code_the_client_got(
    name: str,
) -> None:
    client, extra, key = SCENARIOS[name]()
    with client:
        secret = client.issue_key(**key) if key else client.secret
        response = client.post(
            "/v1/chat/completions", json={**BODY, **extra}, headers=client.as_key(secret)
        )
        [row] = logged(client)

    assert response.status_code >= 400
    assert row.status_code == response.status_code
    assert row.error_code == response.json()["error"].get("code")


def test_an_open_circuit_breaker_is_logged_as_the_503_the_client_got() -> None:
    client = app_client(
        (FakeProvider(fail_first=99), {"name": TEST_ALIAS, **PRICED}),
        breaker_failure_threshold=1,
    )
    with client:
        first, second = chat(client), chat(client)
        rows = logged(client)

    assert (first.status_code, second.status_code) == (502, 503)
    assert [(r.status_code, r.error_code) for r in rows] == [
        (502, "upstream_unavailable"),
        (503, "providers_unavailable"),
    ]


WINDOW = Window(limit=10, remaining=0, reset_seconds=1.0)


def test_the_outcome_mapping_covers_the_errors_that_are_hard_to_provoke() -> None:
    assert outcome_of(UnknownModelError("x")) == (404, "model_not_found")
    assert outcome_of(ContextLengthExceededError(window=1, required=2)) == (
        400,
        "context_length_exceeded",
    )
    assert outcome_of(InvalidApiKeyError()) == (401, "invalid_api_key")
    assert outcome_of(InsufficientScopeError("chat:write")) == (403, "insufficient_scope")
    assert outcome_of(RateLimitExceededError(10, 1.0, WINDOW)) == (429, "rate_limit_exceeded")
    assert outcome_of(TokenLimitExceededError(10, 50, None, WINDOW)) == (429, "request_too_large")
    assert outcome_of(TokenLimitExceededError(10, 5, 1.0, WINDOW)) == (429, "rate_limit_exceeded")
    assert outcome_of(BudgetExceededError("key", 1_000, 1.0)) == (429, "budget_exceeded")
    assert outcome_of(QuotaUnavailableError(1.0)) == (503, "quota_unavailable")
    assert outcome_of(TenantPolicyError("no_model_under_cap", "x")) == (
        403,
        "routing_policy_no_model",
    )
    assert outcome_of(TenantPolicyError("pin_unavailable", "x")) == (503, "tenant_policy_invalid")
    assert outcome_of(StarletteHTTPException(status_code=418)) == (418, None)
    assert outcome_of(asyncio.CancelledError()) == (499, "client_disconnected")
    assert outcome_of(RuntimeError("bug")) == (500, "internal_error")
    assert outcome_of(ProviderDown("x")) == (502, "upstream_unavailable")


# ---------- smart routing ----------


def routed_client(cheap: FakeProvider, smart: FakeProvider) -> AuthedClient:
    return app_client(
        (cheap, {"name": "cheap", "tier": Tier.SLM, "input_cost_per_mtok": 1.0,
                 "output_cost_per_mtok": 2.0}),
        (smart, {"name": "smart", "tier": Tier.FRONTIER, "input_cost_per_mtok": 10.0,
                 "output_cost_per_mtok": 30.0}),
    )  # fmt: skip


def test_an_auto_request_records_how_it_was_routed() -> None:
    with routed_client(FakeProvider(), FakeProvider()) as client:
        chat(client, model="auto", messages=[{"role": "user", "content": "Hi!"}])
        [row] = logged(client)

    assert (row.requested_model, row.served_model) == ("auto", "cheap")
    assert row.routed is True and row.tier == "slm" and row.escalated is False
    assert row.complexity is not None and 0 <= row.complexity <= 1


def test_an_escalation_is_recorded_as_one() -> None:
    with routed_client(FakeProvider(fail_first=99), FakeProvider()) as client:
        response = chat(client, model="auto", messages=[{"role": "user", "content": "Hi!"}])
        [row] = logged(client)

    assert response.status_code == 200
    assert (row.served_model, row.tier, row.escalated) == ("smart", "frontier", True)
    assert row.cost_nano_usd == (10 * 10 + 5 * 30) * 1000  # priced as the model that answered


# ---------- logging never gets in the way ----------


def test_a_request_succeeds_even_if_the_log_cannot_be_written(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Broken:
        def __call__(self) -> None:
            raise RuntimeError("database gone")

    with single() as client:
        client.app.state.request_logger = RequestLogger(Broken())  # type: ignore[arg-type,attr-defined]
        with caplog.at_level("ERROR", logger="app.usage.record"):
            response = chat(client)

    assert response.status_code == 200
    assert any("could not write the request log" in r.message for r in caplog.records)


def test_logging_can_be_switched_off() -> None:
    with app_client((FakeProvider(), {"name": TEST_ALIAS}), request_log_enabled=False) as client:
        assert chat(client).status_code == 200
        assert logged(client) == []


# ---------- streams ----------


class Capture(RequestLogger):
    def __init__(self) -> None:
        super().__init__(None)
        self.entries: list[LogEntry] = []

    async def write(self, entry: LogEntry) -> None:
        self.entries.append(entry)


def new_entry() -> LogEntry:
    return LogEntry(tenant_id="t", key_id="k", requested_model="m", stream=True)


async def test_a_stream_that_finishes_is_logged_once_without_an_error() -> None:
    capture, entry = Capture(), new_entry()
    recorder = Recorder(reserved=500, prompt=8)

    events = [
        e
        async for e in _event_stream(
            text_chunk("x" * 80), chunks(), recorder.quota, Served(), entry, capture
        )
    ]

    assert events[-1] == "data: [DONE]\n\n"
    assert capture.entries == [entry] and entry.error_code is None
    assert entry.prompt_tokens == 8 and entry.completion_tokens > 0


async def test_a_client_that_goes_away_mid_stream_is_logged_as_having_done_so() -> None:
    capture, entry = Capture(), new_entry()
    recorder = Recorder(reserved=500, prompt=8)
    stream = _event_stream(
        text_chunk("a" * 40),
        chunks(text_chunk("b" * 400)),
        recorder.quota,
        Served(),
        entry,
        capture,
    )

    await anext(stream)
    await stream.aclose()

    assert capture.entries == [entry]
    assert entry.error_code == "client_disconnected"


async def test_a_stream_that_breaks_after_it_began_is_logged_with_the_error() -> None:
    async def dying() -> AsyncIterator[Any]:
        raise ProviderDown("upstream died", provider="x")
        yield  # pragma: no cover

    capture, entry = Capture(), new_entry()
    recorder = Recorder(reserved=500, prompt=8)

    [e async for e in _event_stream(
        text_chunk("x" * 40), dying(), recorder.quota, Served(), entry, capture
    )]  # fmt: skip

    assert capture.entries == [entry]
    assert entry.error_code == "upstream_unavailable"
    assert entry.status_code == 200  # the response had already begun


async def test_an_unexpected_error_mid_stream_is_logged_as_internal() -> None:
    async def buggy() -> AsyncIterator[Any]:
        raise RuntimeError("bug")
        yield  # pragma: no cover

    capture, entry = Capture(), new_entry()
    recorder = Recorder(reserved=500, prompt=8)

    [e async for e in _event_stream(
        text_chunk("x" * 40), buggy(), recorder.quota, Served(), entry, capture
    )]  # fmt: skip

    assert entry.error_code == "internal_error"


# ---------- a request does not hold a database connection while the model works ----------


def test_the_database_connection_is_returned_before_the_model_is_called(tmp_path: Path) -> None:
    class Watching(FakeProvider):
        engine: Any = None
        checked_out: list[int]

        async def chat(self, request: Any) -> Any:
            self.checked_out.append(self.engine.pool.checkedout())
            return await super().chat(request)

    provider = Watching()
    provider.checked_out = []
    database = f"sqlite+aiosqlite:///{tmp_path / 'pool.db'}"
    with app_client((provider, {"name": TEST_ALIAS}), database_url=database) as client:
        provider.engine = client.app.state.db_engine  # type: ignore[attr-defined]
        assert chat(client).status_code == 200

    assert provider.checked_out == [0]  # held none while the model was working


async def test_many_slow_requests_at_once_neither_starve_the_pool_nor_lose_logs(
    tmp_path: Path,
) -> None:
    """More requests in flight than the pool has connections, each logging when it finishes."""
    provider = FakeProvider(latency=0.1)

    def factory(_: httpx.AsyncClient) -> ModelRegistry:
        return ModelRegistry([Deployment.create(model_config(name=TEST_ALIAS), provider)])

    database = f"sqlite+aiosqlite:///{tmp_path / 'busy.db'}"
    app = create_app(Settings(_env_file=None, database_url=database, retry_base_delay=0.0), factory)
    async with app.router.lifespan_context(app):
        secret = await seed_credentials(app)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gw",
            headers={"Authorization": f"Bearer {secret}"},
        ) as client:
            responses = await asyncio.wait_for(
                asyncio.gather(
                    *[client.post("/v1/chat/completions", json=BODY) for _ in range(40)]
                ),
                timeout=20,
            )
        async with app.state.db_session_factory() as session:
            count = len((await session.execute(select(RequestLog))).scalars().all())

    assert {r.status_code for r in responses} == {200}
    assert count == 40
