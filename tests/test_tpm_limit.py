import asyncio
import logging
import os
import sqlite3
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from fastapi import FastAPI
from openai import OpenAI, RateLimitError
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from app.api.chat import _event_stream
from app.auth.dependencies import AuthContext
from app.core.config import Settings
from app.core.redis import build_redis_client
from app.providers.errors import ProviderDown
from app.providers.fake import FakeProvider
from app.quota.bucket import BucketStore, TakeResult
from app.quota.limiter import (
    NO_RESERVATION,
    RateLimiter,
    Reservation,
    TokenLimitExceededError,
)
from app.router.tokens import estimate_text_tokens
from app.schemas.chat import ChatCompletionRequest
from app.schemas.completion import ChatCompletionChunk, ChunkChoice, Delta
from tests.conftest import AuthedClient
from tests.test_db import alembic_config
from tests.test_rpm_limit import Clock, chat, make_client

KEY = "tpm-key"


def auth(tpm_limit: int | None = 1_000) -> AuthContext:
    return AuthContext(
        key_id=KEY,
        key_prefix="llmr_abc",
        tenant_id="tenant-1",
        tenant_name="acme",
        scopes=frozenset({"chat:write"}),
        allowed_models=None,
        rpm_limit=None,
        tpm_limit=tpm_limit,
    )


def request(content: str = "Hi", **extra: Any) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": content}], **extra}
    )


# "Hi" costs 8 prompt tokens: 3 for the reply priming, 4 for the message, 1 for its text.
PROMPT_HI = 8


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(redis_client: Redis) -> BucketStore:
    return BucketStore(redis_client)


@pytest.fixture
def limiter(store: BucketStore, clock: Clock) -> RateLimiter:
    return RateLimiter(store, completion_reserve=512, now_ms=clock)


async def balance(store: BucketStore, clock: Clock, limit: int = 1_000) -> int:
    result = await store.take(
        f"tpm:{KEY}", cost=0, capacity=limit, refill_per_second=limit / 60, now_ms=clock.now
    )
    return result.remaining


# ---------- how much is reserved ----------


def test_the_estimate_is_the_prompt_plus_max_tokens(limiter: RateLimiter) -> None:
    assert limiter.estimate_tokens(request(max_tokens=100)) == (PROMPT_HI, PROMPT_HI + 100)


def test_without_max_tokens_a_default_completion_is_assumed(limiter: RateLimiter) -> None:
    assert limiter.estimate_tokens(request()) == (PROMPT_HI, PROMPT_HI + 512)


def test_several_choices_reserve_several_completions(limiter: RateLimiter) -> None:
    assert limiter.estimate_tokens(request(max_tokens=100, n=3)) == (PROMPT_HI, PROMPT_HI + 300)


def test_longer_prompts_reserve_more(limiter: RateLimiter) -> None:
    short = limiter.estimate_tokens(request("hi", max_tokens=10))[1]
    long = limiter.estimate_tokens(request("hello there " * 100, max_tokens=10))[1]

    assert long > short + 200


# ---------- reserving ----------


async def test_a_reservation_takes_its_tokens(
    limiter: RateLimiter, store: BucketStore, clock: Clock
) -> None:
    reservation = await limiter.reserve_tokens(auth(), request(max_tokens=92))

    assert reservation.reserved == 100
    assert reservation.prompt_tokens == PROMPT_HI
    assert await balance(store, clock) == 900


async def test_a_key_cannot_reserve_more_than_it_has(
    limiter: RateLimiter, store: BucketStore, clock: Clock
) -> None:
    await limiter.reserve_tokens(auth(), request(max_tokens=892))  # 900 of 1000

    with pytest.raises(TokenLimitExceededError) as excinfo:
        await limiter.reserve_tokens(auth(), request(max_tokens=292))  # needs 300, 100 left

    error = excinfo.value
    assert (error.limit, error.requested, error.too_large) == (1_000, 300, False)
    assert error.retry_after == pytest.approx(12.0)  # 200 short at 1000/60 a second
    assert "1000 tokens per minute" in str(error)
    assert "12s" in str(error)
    assert await balance(store, clock) == 100  # the refused attempt took nothing


async def test_a_request_bigger_than_the_whole_limit_is_flagged(
    limiter: RateLimiter, store: BucketStore, clock: Clock
) -> None:
    with pytest.raises(TokenLimitExceededError) as excinfo:
        await limiter.reserve_tokens(auth(1_000), request(max_tokens=2_000))

    assert excinfo.value.too_large is True
    assert excinfo.value.retry_after is None
    assert "Request too large" in str(excinfo.value)
    assert await balance(store, clock) == 1_000


async def test_allowance_returns_over_time(limiter: RateLimiter, clock: Clock) -> None:
    await limiter.reserve_tokens(auth(), request(max_tokens=992))  # all 1000
    with pytest.raises(TokenLimitExceededError):
        await limiter.reserve_tokens(auth(), request(max_tokens=92))

    clock.now += 6_000  # 100 tokens' worth at 1000 a minute

    await limiter.reserve_tokens(auth(), request(max_tokens=92))


async def test_keys_have_separate_token_allowances(limiter: RateLimiter) -> None:
    other = AuthContext(**{**auth().__dict__, "key_id": "someone-else"})
    await limiter.reserve_tokens(auth(), request(max_tokens=992))

    await limiter.reserve_tokens(other, request(max_tokens=992))  # unaffected


async def test_without_a_limit_nothing_is_reserved() -> None:
    class Forbidden(BucketStore):
        def __init__(self) -> None:
            pass

        async def take(self, *args: Any, **kwargs: Any) -> TakeResult:
            raise AssertionError("Redis must not be touched when there is no limit")

    reservation = await RateLimiter(Forbidden()).reserve_tokens(auth(None), request())

    assert reservation is NO_RESERVATION
    await reservation.settle(123)  # harmless


async def test_the_default_limit_applies_and_a_keys_own_wins(
    store: BucketStore, clock: Clock
) -> None:
    limiter = RateLimiter(store, default_tpm=300, now_ms=clock)

    with pytest.raises(TokenLimitExceededError):
        await limiter.reserve_tokens(auth(None), request())  # 520 > default 300
    own = AuthContext(**{**auth(5_000).__dict__, "key_id": "another-key"})
    await limiter.reserve_tokens(own, request())  # its own, larger limit


async def test_concurrent_reservations_never_oversubscribe(
    limiter: RateLimiter, store: BucketStore, clock: Clock
) -> None:
    async def attempt() -> bool:
        try:
            await limiter.reserve_tokens(auth(), request(max_tokens=92))  # 100 each
        except TokenLimitExceededError:
            return False
        return True

    outcomes = await asyncio.gather(*(attempt() for _ in range(50)))

    assert sum(outcomes) == 10
    assert await balance(store, clock) == 0


# ---------- settling ----------


async def test_unused_tokens_are_refunded(
    limiter: RateLimiter, store: BucketStore, clock: Clock
) -> None:
    reservation = await limiter.reserve_tokens(auth(), request(max_tokens=492))  # 500

    await reservation.settle(120)

    assert await balance(store, clock) == 880  # only 120 were really used


async def test_extra_usage_is_charged(
    limiter: RateLimiter, store: BucketStore, clock: Clock
) -> None:
    reservation = await limiter.reserve_tokens(auth(), request(max_tokens=92))  # 100

    await reservation.settle(350)

    assert await balance(store, clock) == 650


async def test_exact_usage_changes_nothing(
    limiter: RateLimiter, store: BucketStore, clock: Clock
) -> None:
    reservation = await limiter.reserve_tokens(auth(), request(max_tokens=92))

    await reservation.settle(100)

    assert await balance(store, clock) == 900


async def test_a_reservation_settles_only_once(
    limiter: RateLimiter, store: BucketStore, clock: Clock
) -> None:
    reservation = await limiter.reserve_tokens(auth(), request(max_tokens=92))

    await reservation.settle(40)
    await reservation.settle(40)
    await reservation.release()

    assert await balance(store, clock) == 960


async def test_releasing_refunds_everything(
    limiter: RateLimiter, store: BucketStore, clock: Clock
) -> None:
    reservation = await limiter.reserve_tokens(auth(), request(max_tokens=492))

    await reservation.release()

    assert await balance(store, clock) == 1_000


async def test_negative_usage_is_treated_as_zero(
    limiter: RateLimiter, store: BucketStore, clock: Clock
) -> None:
    reservation = await limiter.reserve_tokens(auth(), request(max_tokens=92))

    await reservation.settle(-50)

    assert await balance(store, clock) == 1_000


async def test_an_overrun_leaves_a_debt_that_blocks_the_next_request(
    limiter: RateLimiter, store: BucketStore, clock: Clock
) -> None:
    reservation = await limiter.reserve_tokens(auth(), request(max_tokens=892))  # 900
    await reservation.settle(1_500)  # used far more than reserved

    assert await balance(store, clock) == -500
    with pytest.raises(TokenLimitExceededError) as excinfo:
        await limiter.reserve_tokens(auth(), request(max_tokens=92))  # 100: needs 600 of refill
    assert excinfo.value.retry_after == pytest.approx(36.0)  # 600 at 1000 a minute
    clock.now += 36_000
    await limiter.reserve_tokens(auth(), request(max_tokens=92))  # the debt is paid off


# ---------- when Redis misbehaves ----------


class BrokenStore(BucketStore):
    def __init__(self, error: Exception, fail_adjust_only: bool = False) -> None:
        self._error = error
        self._fail_adjust_only = fail_adjust_only
        self.balance = 1_000

    async def take(self, *args: Any, **kwargs: Any) -> TakeResult:
        if not self._fail_adjust_only:
            raise self._error
        return TakeResult(True, 900, 0.0)

    async def adjust(self, *args: Any, **kwargs: Any) -> int:
        raise self._error


@pytest.mark.parametrize(
    "error", [RedisConnectionError("down"), TimeoutError("slow"), ConnectionResetError("reset")]
)
async def test_reserving_fails_open(error: Exception, caplog: pytest.LogCaptureFixture) -> None:
    limiter = RateLimiter(BrokenStore(error))

    with caplog.at_level(logging.WARNING, logger="app.quota.limiter"):
        reservation = await limiter.reserve_tokens(auth(), request())

    assert reservation is NO_RESERVATION
    assert any("token limiting unavailable" in r.getMessage() for r in caplog.records)


async def test_a_failed_settlement_is_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    limiter = RateLimiter(BrokenStore(RedisConnectionError("down"), fail_adjust_only=True))
    reservation = await limiter.reserve_tokens(auth(), request(max_tokens=92))

    with caplog.at_level(logging.WARNING, logger="app.quota.limiter"):
        await reservation.settle(10)  # must not blow up the response that was already produced

    assert any("could not settle" in r.getMessage() for r in caplog.records)


# ---------- the stream wrapper (disconnects, failures) ----------


def text_chunk(text: str) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="c", created=0, model="m", choices=[ChunkChoice(index=0, delta=Delta(content=text))]
    )


class Recorder:
    def __init__(self, reserved: int = 500, prompt: int = 8) -> None:
        self.settled_with: list[int] = []

        async def adjust(delta: int) -> None:
            self.settled_with.append(reserved - delta)  # recover the "actual" from the delta

        self.reservation = Reservation(reserved, prompt, adjust)


async def chunks(*items: ChatCompletionChunk) -> AsyncIterator[ChatCompletionChunk]:
    for item in items:
        yield item


async def test_a_client_disconnecting_mid_stream_still_settles() -> None:
    recorder = Recorder(reserved=500, prompt=8)
    first = text_chunk("a" * 40)  # about 10 tokens of text
    stream = _event_stream(first, chunks(text_chunk("b" * 400)), recorder.reservation)

    await anext(stream)  # the client reads one event ...
    await stream.aclose()  # ... then goes away

    assert recorder.settled_with == [8 + estimate_text_tokens("a" * 40)]


async def test_a_stream_that_runs_dry_settles_on_what_was_sent() -> None:
    recorder = Recorder(reserved=500, prompt=8)

    events = [e async for e in _event_stream(text_chunk("x" * 80), chunks(), recorder.reservation)]

    assert events[-1] == "data: [DONE]\n\n"
    assert recorder.settled_with == [8 + 20]


async def test_reported_usage_beats_the_estimate() -> None:
    recorder = Recorder(reserved=500, prompt=8)
    usage = ChatCompletionChunk.model_validate(
        {
            "id": "c",
            "created": 0,
            "model": "m",
            "choices": [],
            "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
        }
    )

    _ = [e async for e in _event_stream(text_chunk("x" * 400), chunks(usage), recorder.reservation)]

    assert recorder.settled_with == [42]


async def test_a_stream_that_fails_part_way_settles_on_what_was_sent() -> None:
    recorder = Recorder(reserved=500, prompt=8)

    async def dying() -> AsyncIterator[ChatCompletionChunk]:
        yield text_chunk("y" * 40)
        raise ProviderDown("reset", provider="p")

    events = [e async for e in _event_stream(text_chunk("x" * 40), dying(), recorder.reservation)]

    assert "upstream_unavailable" in events[-1]
    assert recorder.settled_with == [8 + 20]  # charged for the 20 tokens that were delivered


# ---------- over HTTP ----------

PROMPT_AND_DEFAULT_RESERVE = PROMPT_HI + 512


def bucket_tokens(client: AuthedClient) -> float:
    app: FastAPI = client.app  # type: ignore[assignment]
    assert client.portal is not None

    async def read() -> list[float]:
        redis = app.state.redis
        return [float(await redis.hget(key, "tokens")) async for key in redis.scan_iter("*:tpm:*")]

    found: list[float] = client.portal.call(read)
    assert len(found) == 1, found
    return found[0]


def test_unused_reservations_are_given_back_so_normal_traffic_flows() -> None:
    with make_client() as client:  # each reply uses 15 tokens, but 520 are reserved up front
        secret = client.issue_key(tpm_limit=1_000)

        statuses = [chat(client, secret).status_code for _ in range(20)]

        remaining = bucket_tokens(client)

    assert statuses == [200] * 20  # without settling, the second request would already fail
    assert 690 <= remaining <= 715  # 1000 minus 20 * 15, plus a little refill


def test_overspending_is_charged_and_slows_the_next_request() -> None:
    heavy = FakeProvider(prompt_tokens=400, completion_tokens=400)  # 800 used per call
    with make_client(heavy) as client:
        secret = client.issue_key(tpm_limit=1_000)

        first = chat(client, secret)
        second = chat(client, secret)

        left = bucket_tokens(client)

    assert first.status_code == 200
    assert 195 <= left <= 215  # 1000 - 800 used
    assert second.status_code == 429
    error = second.json()["error"]
    assert error["code"] == "rate_limit_exceeded"
    assert error["type"] == "rate_limit_error"
    assert "tokens per minute" in error["message"]
    assert 15 <= int(second.headers["Retry-After"]) <= 21  # 320 short at 1000 a minute
    assert heavy.calls == 1


def test_a_request_too_big_for_the_key_is_refused_without_retry_advice() -> None:
    provider = FakeProvider()
    with make_client(provider) as client:
        secret = client.issue_key(tpm_limit=300)

        default_reserve = chat(client, secret)  # reserves ~520 > 300
        explicit = chat(client, secret, max_tokens=5_000)

    for response in (default_reserve, explicit):
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "request_too_large"
        assert "Retry-After" not in response.headers
    assert provider.calls == 0


def test_a_smaller_request_fits_where_a_bigger_one_did_not() -> None:
    with make_client() as client:
        secret = client.issue_key(tpm_limit=300)

        assert chat(client, secret).status_code == 429
        assert chat(client, secret, max_tokens=50).status_code == 200


def test_failed_requests_give_everything_back() -> None:
    provider = FakeProvider(fail_first=99)
    with make_client(provider, retry_max_attempts=1) as client:
        secret = client.issue_key(tpm_limit=1_000)

        failures = [chat(client, secret).status_code for _ in range(10)]  # 5,200 reserved, in all
        provider.fail_first = provider.calls  # the vendor recovers
        recovered = chat(client, secret).status_code

    assert failures == [502] * 10  # never 429: each reservation was returned
    assert recovered == 200


def test_requests_for_unknown_models_give_everything_back() -> None:
    with make_client() as client:
        secret = client.issue_key(tpm_limit=1_000)

        missing = [chat(client, secret, model="nope").status_code for _ in range(10)]
        ok = chat(client, secret).status_code

    assert missing == [404] * 10
    assert ok == 200


def test_requests_for_forbidden_models_reserve_nothing() -> None:
    with make_client() as client:
        secret = client.issue_key(tpm_limit=1_000, allowed_models=["something-else"])

        statuses = [chat(client, secret).status_code for _ in range(5)]
        app: FastAPI = client.app  # type: ignore[assignment]
        assert client.portal is not None
        keys = client.portal.call(lambda: app.state.redis.keys("*:tpm:*"))

    assert statuses == [403] * 5
    assert keys == []  # not even a bucket was created


def test_streams_are_settled_from_the_reported_usage() -> None:
    provider = FakeProvider(stream_usage=True, prompt_tokens=30, completion_tokens=20)
    with make_client(provider) as client:
        secret = client.issue_key(tpm_limit=1_000)

        statuses = [chat(client, secret, stream=True).status_code for _ in range(10)]
        remaining = bucket_tokens(client)

    assert statuses == [200] * 10
    assert 495 <= remaining <= 520  # 1000 - 10 * 50, plus a little refill


def test_streams_without_usage_are_settled_from_the_text_sent() -> None:
    reply = "one two three four five six seven eight nine ten"  # 47 characters: 12 tokens
    provider = FakeProvider(reply=reply)  # reports no usage when streaming
    with make_client(provider) as client:
        secret = client.issue_key(tpm_limit=1_000)

        statuses = [chat(client, secret, stream=True).status_code for _ in range(10)]
        remaining = bucket_tokens(client)

    assert statuses == [200] * 10
    assert 795 <= remaining <= 815  # 10 * (8 + 12) tokens used


def test_a_stream_that_fails_to_start_gives_everything_back() -> None:
    provider = FakeProvider(fail_first=99)
    with make_client(provider, retry_max_attempts=1) as client:
        secret = client.issue_key(tpm_limit=1_000)

        failures = [chat(client, secret, stream=True).status_code for _ in range(10)]

    assert failures == [502] * 10


def test_limits_are_per_key() -> None:
    heavy = FakeProvider(prompt_tokens=400, completion_tokens=400)
    with make_client(heavy) as client:
        spent = client.issue_key(tpm_limit=1_000)
        fresh = client.issue_key(tpm_limit=1_000)
        chat(client, spent)
        assert chat(client, spent).status_code == 429

        assert chat(client, fresh).status_code == 200


def test_the_gateway_default_and_a_per_key_override() -> None:
    with make_client(default_tpm_limit=300) as client:
        generous = client.issue_key(tpm_limit=50_000)

        assert chat(client, client.secret).status_code == 429  # the default applies
        assert chat(client, generous).status_code == 200  # its own limit wins


def test_the_allowance_comes_back_with_time() -> None:
    heavy = FakeProvider(prompt_tokens=400, completion_tokens=400)
    with make_client(heavy) as client:
        secret = client.issue_key(tpm_limit=1_000)
        chat(client, secret)
        assert chat(client, secret).status_code == 429
        app: FastAPI = client.app  # type: ignore[assignment]

        async def a_minute_passes() -> None:
            async for bucket in app.state.redis.scan_iter("*:tpm:*"):
                stamp = int(await app.state.redis.hget(bucket, "ts"))
                await app.state.redis.hset(bucket, "ts", stamp - 61_000)

        assert client.portal is not None
        client.portal.call(a_minute_passes)

        assert chat(client, secret).status_code == 200


def test_the_sdk_sees_a_rate_limit_error() -> None:
    with make_client() as client:
        secret = client.issue_key(tpm_limit=300)
        sdk = OpenAI(
            base_url="http://testserver/v1", api_key=secret, http_client=client, max_retries=0
        )

        with pytest.raises(RateLimitError) as excinfo:
            sdk.chat.completions.create(
                model="test-model", messages=[{"role": "user", "content": "hi"}]
            )

    assert excinfo.value.code == "request_too_large"


def test_requests_still_work_when_redis_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    unreachable = Settings(_env_file=None, redis_url="redis://127.0.0.1:1/0", redis_timeout=0.2)
    monkeypatch.setattr(
        "app.lifespan.build_redis_client", lambda _: build_redis_client(unreachable)
    )

    with make_client() as client:
        secret = client.issue_key(tpm_limit=300)

        statuses = [chat(client, secret).status_code for _ in range(3)]

    assert statuses == [200] * 3  # fails open


@pytest.mark.parametrize("backend", ["fake", "real"])
def test_concurrent_requests_through_the_whole_stack_cannot_exceed_the_allowance(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """12 slow requests, each reserving 100 tokens against a limit of 500, all in flight at once."""
    if backend == "real":
        url = os.environ.get("LLMROUTER_TEST_REDIS_URL")
        if not url:
            pytest.skip("set LLMROUTER_TEST_REDIS_URL to test against a real Redis")
        real = Settings(_env_file=None, redis_url=url)
        monkeypatch.setattr("app.lifespan.build_redis_client", lambda _: build_redis_client(real))

    slow = FakeProvider(latency=0.4)  # keeps every reservation outstanding while the others arrive
    with make_client(slow) as client:
        app: FastAPI = client.app  # type: ignore[assignment]
        assert client.portal is not None
        client.portal.call(app.state.redis.flushdb)
        secret = client.issue_key(tpm_limit=500)

        with ThreadPoolExecutor(max_workers=12) as pool:
            statuses = list(
                pool.map(lambda _: chat(client, secret, max_tokens=92).status_code, range(12))
            )

        client.portal.call(app.state.redis.flushdb)

    assert statuses.count(200) == 5  # 5 * 100 reserved = 500; the other 7 are refused
    assert statuses.count(429) == 7


# ---------- configuring limits ----------


def test_the_admin_api_sets_shows_and_inherits_the_limit() -> None:
    with make_client() as client:
        admin = client.as_key(client.issue_key(["admin:tenants", "admin:keys"]))
        tenant = client.post("/admin/tenants", json={"name": "globex"}, headers=admin).json()["id"]
        base = f"/admin/tenants/{tenant}/keys"

        created = client.post(base, json={"name": "app", "tpm_limit": 5_000}, headers=admin).json()
        default = client.post(base, json={"name": "other"}, headers=admin).json()
        kept = client.post(f"{base}/{created['id']}/rotate", json={}, headers=admin).json()
        changed = client.post(
            f"{base}/{kept['id']}/rotate", json={"tpm_limit": 100}, headers=admin
        ).json()
        reset = client.post(
            f"{base}/{changed['id']}/rotate", json={"tpm_limit": None}, headers=admin
        ).json()

    assert (created["tpm_limit"], default["tpm_limit"]) == (5_000, None)
    assert (kept["tpm_limit"], changed["tpm_limit"], reset["tpm_limit"]) == (5_000, 100, None)


@pytest.mark.parametrize("value", [0, -1, 1_000_000_001, "lots"])
def test_invalid_token_limits_are_rejected(value: Any) -> None:
    with make_client() as client:
        admin = client.as_key(client.issue_key(["admin:tenants", "admin:keys"]))
        tenant = client.post("/admin/tenants", json={"name": "globex"}, headers=admin).json()["id"]

        response = client.post(
            f"/admin/tenants/{tenant}/keys", json={"name": "app", "tpm_limit": value}, headers=admin
        )

    assert response.status_code == 400


def test_migration_leaves_existing_keys_on_the_default(tmp_path: Path) -> None:
    database = tmp_path / "tpm.db"
    config = alembic_config(database)
    command.upgrade(config, "0005")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO tenants (id, name, is_active, created_at)"
            " VALUES ('t1', 'acme', 1, '2026-01-01')"
        )
        connection.execute(
            "INSERT INTO api_keys (id, tenant_id, name, key_prefix, key_hash, created_at, scopes)"
            " VALUES ('k1', 't1', 'old key', 'llmr_abc', 'hash', '2026-01-01', '[]')"
        )

    command.upgrade(config, "head")

    with sqlite3.connect(database) as connection:
        (stored,) = connection.execute("SELECT tpm_limit FROM api_keys WHERE id = 'k1'").fetchone()
    assert stored is None


def test_the_token_limit_migration_can_be_reversed(tmp_path: Path) -> None:
    database = tmp_path / "tpm.db"
    config = alembic_config(database)
    command.upgrade(config, "head")

    command.downgrade(config, "0005")

    with sqlite3.connect(database) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(api_keys)")]
    assert "tpm_limit" not in columns
    assert "rpm_limit" in columns
