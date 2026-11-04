"""The quota guarantees under real concurrency, across several gateway replicas.

Each test starts N independent app instances (separate event loops, as separate processes would
be) that share one database file and one Redis, then fires many requests at all of them at once.
However the requests interleave, the limits must hold exactly.
"""

import asyncio
import os
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fakeredis
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import update

from app.auth.service import create_api_key
from app.core.config import Settings
from app.core.redis import build_redis_client
from app.db.base import Base
from app.db.models import ApiKey, Tenant, utcnow
from app.main import create_app
from app.providers.fake import FakeProvider
from app.quota.budget import month_period, usd_to_nano
from app.router.deployment import Deployment
from app.router.registry import ModelRegistry
from tests.conftest import model_config

LIMIT = 10
REQUESTS = 100
PRICES: dict[str, Any] = {"input_cost_per_mtok": 100.0, "output_cost_per_mtok": 200.0}
# Each reply uses 10 prompt + 5 completion tokens: $0.002 at the prices above.
COST_PER_REPLY = usd_to_nano(0.002)
# "Hi" is 8 prompt tokens, so max_tokens=92 reserves 100 tokens and max_tokens=5 reserves $0.0018.
RESERVED_USD = (8 * 100 + 5 * 200) / 1_000_000


@dataclass
class Replicas:
    clients: list[TestClient]
    apps: list[FastAPI]
    providers: list[FakeProvider]
    tenant_id: str
    gate: threading.Event | None = None  # see `replicas(hold=True)`
    _keys: dict[str, str] = field(default_factory=dict)

    @property
    def redis(self) -> Any:
        return self.apps[0].state.redis

    def call(self, work: Callable[[Any], Any]) -> Any:
        """Run async work against the shared database, on the first replica's event loop."""
        portal = self.clients[0].portal
        assert portal is not None

        async def run() -> Any:
            async with self.apps[0].state.db_session_factory() as session:
                return await work(session)

        return portal.call(run)

    def issue_key(self, **options: Any) -> tuple[str, str]:
        """Create a key (returning its secret and id) in the shared database."""

        async def create(session: Any) -> tuple[str, str]:
            key, secret = await create_api_key(session, self.tenant_id, "load test", **options)
            # Recently used, so requests do not all try to write last_used_at at once.
            await session.execute(
                update(ApiKey).where(ApiKey.id == key.id).values(last_used_at=utcnow())
            )
            await session.commit()
            return secret, key.id

        secret, key_id = self.call(create)
        return secret, key_id

    def set_tenant_budget(self, usd: float) -> None:
        async def apply(session: Any) -> None:
            await session.execute(
                update(Tenant)
                .where(Tenant.id == self.tenant_id)
                .values(monthly_budget_nano_usd=usd_to_nano(usd))
            )
            await session.commit()

        self.call(apply)

    def fire(
        self,
        secret: str | list[str],
        count: int = REQUESTS,
        **body: Any,
    ) -> list[tuple[int, httpx.Response]]:
        """Send ``count`` requests at once, spread over every replica and (if several) key."""
        secrets = [secret] if isinstance(secret, str) else secret

        def one(index: int) -> tuple[int, httpx.Response]:
            client = self.clients[index % len(self.clients)]
            key = index % len(secrets)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                    **body,
                },
                headers={"Authorization": f"Bearer {secrets[key]}"},
            )
            return key, response

        with ThreadPoolExecutor(max_workers=count) as pool:  # one thread each: all at once
            futures = [pool.submit(one, index) for index in range(count)]
            if self.gate is not None:
                self._release_when_decided(futures, count)
            return [future.result() for future in futures]

    def _release_when_decided(self, futures: list[Any], count: int) -> None:
        """Let the admitted requests finish only once every request has been decided.

        Each request is either refused (and has already answered) or admitted (and is waiting at
        its provider). When those two add up to the whole burst, every decision has been made, and
        none of them could have been affected by a reply finishing and handing back what it did
        not use. That is the situation these tests describe, and holding the replies is how they
        get it on a machine of any speed instead of hoping a fixed delay is long enough.
        """
        assert self.gate is not None
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            answered = sum(1 for future in futures if future.done())
            waiting = sum(provider.calls for provider in self.providers)
            if answered + waiting >= count:
                break
            time.sleep(0.01)
        self.gate.set()

    async def hash_field(self, name: str, field_name: str) -> float:
        return float(await self.redis.hget(name, field_name))


@contextmanager
def replicas(
    count: int,
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    latency: float = 0.0,
    hold: bool = False,
    **settings: Any,
) -> Iterator[Replicas]:
    """Start ``count`` replicas that share a database and a Redis.

    ``hold`` makes every provider wait, after it has been called, until the test says the burst
    has been decided (see ``Replicas.fire``), instead of for a fixed time.
    """
    gate = threading.Event() if hold else None
    monkeypatch.setenv("LLMROUTER_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/shared.db")
    if backend == "real":
        url = os.environ.get("LLMROUTER_TEST_REDIS_URL")
        if not url:
            pytest.skip("set LLMROUTER_TEST_REDIS_URL to test against a real Redis")
        real = Settings(_env_file=None, redis_url=url)
        monkeypatch.setattr("app.lifespan.build_redis_client", lambda _: build_redis_client(real))
    else:
        server = fakeredis.FakeServer()  # one "Redis" that every replica talks to
        monkeypatch.setattr(
            "app.lifespan.build_redis_client",
            lambda _: fakeredis.FakeAsyncRedis(server=server, decode_responses=True),
        )

    providers: list[FakeProvider] = []

    async def wait_for_gate(_: float) -> None:
        """Wait for the gate without a thread. Parking each held request in the event loop's
        thread pool would starve it on a small machine (it has cpu_count + 4 threads), and
        connecting to a real Redis needs that pool: connections would time out and the quota
        guard, as designed, would fail open and wave requests through."""
        assert gate is not None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 120.0
        # Polling is deliberate: the gate is shared by several event loops (one per replica),
        # which an asyncio or anyio Event cannot be.
        while not gate.is_set() and loop.time() < deadline:  # noqa: ASYNC110
            await asyncio.sleep(0.01)

    def registry_factory(_: httpx.AsyncClient) -> ModelRegistry:
        if gate is not None:
            provider = FakeProvider(latency=1.0, sleep=wait_for_gate)  # the delay is the gate
        else:
            provider = FakeProvider(latency=latency)
        providers.append(provider)
        return ModelRegistry([Deployment.create(model_config(**PRICES), provider)])

    options = {"retry_base_delay": 0.0, "retry_max_attempts": 1, **settings}
    apps = [create_app(Settings(_env_file=None, **options), registry_factory) for _ in range(count)]

    with ExitStack() as stack:
        clients = [stack.enter_context(TestClient(app)) for app in apps]
        first = clients[0].portal
        assert first is not None

        async def prepare() -> str:
            await apps[0].state.redis.flushdb()
            async with apps[0].state.db_engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
            async with apps[0].state.db_session_factory() as session:
                tenant = Tenant(name="load-test")
                session.add(tenant)
                await session.commit()
                return tenant.id

        tenant_id = first.call(prepare)
        yield Replicas(clients, apps, providers, tenant_id, gate)
        first.call(apps[0].state.redis.flushdb)


BACKENDS = ["fake", "real"]
REPLICA_COUNTS = [1, 3]


def count(results: list[tuple[int, httpx.Response]], status: int) -> int:
    return sum(1 for _, response in results if response.status_code == status)


def calls(world: Replicas) -> int:
    return sum(provider.calls for provider in world.providers)


# ---------- requests per minute ----------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("replica_count", REPLICA_COUNTS)
def test_one_hundred_requests_against_a_limit_of_ten_let_exactly_ten_through(
    backend: str, replica_count: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with replicas(replica_count, backend, tmp_path, monkeypatch) as world:
        secret, _ = world.issue_key(rpm_limit=LIMIT)

        started = time.monotonic()
        results = world.fire(secret)
        elapsed = time.monotonic() - started

        allowed = count(results, 200)
        # The bucket also refills at LIMIT/60 a second, so allow for that if the box was slow.
        assert LIMIT <= allowed <= LIMIT + int(elapsed // 6)
        assert allowed + count(results, 429) == REQUESTS  # nothing else: no 5xx, no surprises
        assert calls(world) == allowed  # refused requests never reached a provider


@pytest.mark.parametrize("backend", BACKENDS)
def test_refused_requests_are_told_when_to_come_back(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with replicas(3, backend, tmp_path, monkeypatch) as world:
        secret, _ = world.issue_key(rpm_limit=LIMIT)

        results = world.fire(secret)

    refused = [response for _, response in results if response.status_code == 429]
    assert len(refused) >= REQUESTS - LIMIT - 2
    assert all(int(r.headers["Retry-After"]) >= 1 for r in refused)
    assert all(r.json()["error"]["code"] == "rate_limit_exceeded" for r in refused)
    assert all(r.headers["x-ratelimit-remaining-requests"] == "0" for r in refused)


@pytest.mark.parametrize("backend", BACKENDS)
def test_streaming_requests_are_held_to_the_same_limit(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with replicas(3, backend, tmp_path, monkeypatch) as world:
        secret, _ = world.issue_key(rpm_limit=LIMIT)

        started = time.monotonic()
        results = world.fire(secret, stream=True)
        elapsed = time.monotonic() - started

    assert LIMIT <= count(results, 200) <= LIMIT + int(elapsed // 6)
    assert count(results, 200) + count(results, 429) == REQUESTS


@pytest.mark.parametrize("backend", BACKENDS)
def test_many_keys_are_each_limited_independently_under_load(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with replicas(3, backend, tmp_path, monkeypatch) as world:
        secrets = [world.issue_key(rpm_limit=LIMIT)[0] for _ in range(4)]

        started = time.monotonic()
        results = world.fire(secrets, count=4 * REQUESTS)
        elapsed = time.monotonic() - started

    slack = int(elapsed // 6)
    for key in range(4):
        allowed = sum(1 for k, r in results if k == key and r.status_code == 200)
        assert LIMIT <= allowed <= LIMIT + slack, f"key {key} got {allowed}"
    assert count(results, 200) + count(results, 429) == 4 * REQUESTS


# ---------- tokens per minute ----------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("replica_count", REPLICA_COUNTS)
def test_the_token_allowance_is_never_oversubscribed(
    backend: str, replica_count: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each request reserves 100 of 1,000 tokens; the provider's delay keeps them all in flight."""
    with replicas(replica_count, backend, tmp_path, monkeypatch, hold=True) as world:
        secret, _ = world.issue_key(tpm_limit=1_000)

        started = time.monotonic()
        results = world.fire(secret, max_tokens=92)
        elapsed = time.monotonic() - started

        # The bucket also refills (1,000 a minute is one reservation every six seconds), so a
        # slow machine may let one more through; it must never be more than that.
        allowed = count(results, 200)
        assert LIMIT <= allowed <= LIMIT + int(elapsed // 6)
        assert count(results, 429) == REQUESTS - allowed
        assert calls(world) == allowed


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_token_reservation_is_settled_afterwards(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After the burst, only what the ten real replies used is gone: refusals leaked nothing."""
    with replicas(3, backend, tmp_path, monkeypatch, hold=True) as world:
        secret, key_id = world.issue_key(tpm_limit=1_000)

        started = time.monotonic()
        results = world.fire(secret, max_tokens=92)

        portal = world.clients[0].portal
        assert portal is not None
        tokens = portal.call(world.hash_field, f"llmrouter:quota:tpm:{key_id}", "tokens")
        elapsed = time.monotonic() - started

    # 1,000 minus a reply of 15 tokens for each one admitted, plus refill (1,000 a minute) for
    # as long as the test took. Unsettled reservations would leave it near zero; leaked refusals
    # would leave it lower than this.
    admitted = count(results, 200)
    assert admitted >= LIMIT
    expected = 1_000 - 15 * admitted
    assert expected - 10 <= tokens <= expected + 17 * elapsed + 10


# ---------- money ----------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("replica_count", REPLICA_COUNTS)
def test_the_key_budget_is_never_overspent(
    backend: str, replica_count: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The budget covers exactly ten worst-case reservations of $0.0018."""
    with replicas(replica_count, backend, tmp_path, monkeypatch, hold=True) as world:
        secret, key_id = world.issue_key(monthly_budget_usd=LIMIT * RESERVED_USD)

        results = world.fire(secret, max_tokens=5)

        portal = world.clients[0].portal
        assert portal is not None
        period = month_period(datetime.now(UTC))
        spent = int(portal.call(world.redis.get, f"llmrouter:budget:key:{key_id}:{period}"))

    assert count(results, 200) == LIMIT
    assert count(results, 429) == REQUESTS - LIMIT
    assert all(
        r.json()["error"]["code"] == "budget_exceeded" for _, r in results if r.status_code == 429
    )
    assert spent == LIMIT * COST_PER_REPLY  # the ten replies' real cost, to the nano-dollar
    assert calls(world) == LIMIT


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_tighter_of_key_and_tenant_budget_wins_and_both_stay_in_step(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with replicas(3, backend, tmp_path, monkeypatch, hold=True) as world:
        secret, key_id = world.issue_key(monthly_budget_usd=LIMIT * RESERVED_USD)
        world.set_tenant_budget(6 * RESERVED_USD)  # the organisation can afford only six

        results = world.fire(secret, max_tokens=5)

        portal = world.clients[0].portal
        assert portal is not None
        period = month_period(datetime.now(UTC))
        key_spent = int(portal.call(world.redis.get, f"llmrouter:budget:key:{key_id}:{period}"))
        tenant_spent = int(
            portal.call(world.redis.get, f"llmrouter:budget:tenant:{world.tenant_id}:{period}")
        )

    assert count(results, 200) == 6
    assert key_spent == tenant_spent == 6 * COST_PER_REPLY  # all or nothing: never out of step


@pytest.mark.parametrize("backend", BACKENDS)
def test_one_tenants_budget_is_shared_by_all_its_keys_under_load(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with replicas(3, backend, tmp_path, monkeypatch, hold=True) as world:
        secrets = [world.issue_key()[0] for _ in range(5)]  # five keys, none capped itself
        world.set_tenant_budget(8 * RESERVED_USD)

        results = world.fire(secrets, max_tokens=5)

    assert count(results, 200) == 8  # not 8 per key: one budget for the whole organisation
    assert count(results, 429) == REQUESTS - 8


# ---------- everything together ----------


@pytest.mark.parametrize("backend", BACKENDS)
def test_all_the_limits_together_still_let_exactly_the_tightest_through(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requests, tokens and money are all limited at once; whichever is tightest governs."""
    with replicas(3, backend, tmp_path, monkeypatch, hold=True) as world:
        secret, _ = world.issue_key(
            rpm_limit=50,  # fifty requests
            tpm_limit=400,  # 400 tokens: thirty reservations of 13 (max_tokens=5)
            monthly_budget_usd=LIMIT * RESERVED_USD,  # money for ten: the tightest
        )

        results = world.fire(secret, max_tokens=5)

    assert count(results, 200) == LIMIT
    refusals = {r.json()["error"]["code"] for _, r in results if r.status_code == 429}
    assert refusals <= {"budget_exceeded", "rate_limit_exceeded"}
    assert "budget_exceeded" in refusals


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_allowance_is_left_stuck_after_a_storm_of_failures(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream failures give everything back, so a failing provider cannot drain a key."""
    with replicas(3, backend, tmp_path, monkeypatch) as world:
        for provider in world.providers:
            provider.fail_first = 10**6  # every upstream call fails
        secret, key_id = world.issue_key(tpm_limit=1_000, monthly_budget_usd=1.0)

        results = world.fire(secret, max_tokens=92)

        portal = world.clients[0].portal
        assert portal is not None
        tokens = portal.call(world.hash_field, f"llmrouter:quota:tpm:{key_id}", "tokens")
        period = month_period(datetime.now(UTC))
        spent = portal.call(world.redis.get, f"llmrouter:budget:key:{key_id}:{period}")

    # Reservations held by requests still in flight can legitimately crowd out late arrivals
    # (a 429), and once the circuit breakers open the rest are refused outright (a 503), but
    # nothing may succeed, and nothing may be left behind afterwards.
    assert count(results, 502) >= 1
    assert count(results, 502) + count(results, 503) + count(results, 429) == REQUESTS
    assert tokens >= 999  # every reservation of 100 tokens, refused or failed, came back
    assert int(spent) == 0  # and so did all the reserved money
