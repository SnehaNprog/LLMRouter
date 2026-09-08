import asyncio
import random

import pytest
from redis.asyncio import Redis

from app.quota.bucket import BucketStore, TakeResult

T0 = 1_700_000_000_000  # an arbitrary fixed "now", in ms


@pytest.fixture
def store(redis_client: Redis) -> BucketStore:
    return BucketStore(redis_client)


async def take(store: BucketStore, cost: int = 1, **kwargs: object) -> TakeResult:
    options: dict[str, object] = {"capacity": 10, "now_ms": T0, **kwargs}
    return await store.take("b", cost=cost, **options)  # type: ignore[arg-type]


# ---------- the basic contract ----------


async def test_a_new_bucket_starts_full(store: BucketStore) -> None:
    result = await take(store, cost=0)

    assert result == TakeResult(allowed=True, remaining=10, retry_after=0.0)


async def test_taking_reduces_the_balance(store: BucketStore) -> None:
    first = await take(store, 3)
    second = await take(store, 4)

    assert (first.allowed, first.remaining) == (True, 7)
    assert (second.allowed, second.remaining) == (True, 3)


async def test_the_bucket_can_be_emptied_exactly(store: BucketStore) -> None:
    result = await take(store, 10)

    assert (result.allowed, result.remaining) == (True, 0)
    assert (await take(store, 1)).allowed is False


async def test_a_refused_take_spends_nothing(store: BucketStore) -> None:
    await take(store, 8)

    refused = await take(store, 5)  # only 2 left

    assert (refused.allowed, refused.remaining) == (False, 2)
    assert (await take(store, 2)).allowed is True  # the 2 are still there


async def test_cost_zero_reads_without_spending(store: BucketStore) -> None:
    await take(store, 6)

    for _ in range(3):
        assert (await take(store, 0)).remaining == 4


async def test_a_cost_above_capacity_can_never_succeed(store: BucketStore) -> None:
    result = await take(store, 11, refill_per_second=100)

    assert result == TakeResult(allowed=False, remaining=10, retry_after=None)
    assert (await take(store, 10)).allowed is True  # and it took nothing


async def test_buckets_are_independent(store: BucketStore) -> None:
    await store.take("a", cost=10, capacity=10, now_ms=T0)

    other = await store.take("b", cost=1, capacity=10, now_ms=T0)

    assert other.allowed is True
    assert (await store.take("a", cost=1, capacity=10, now_ms=T0)).allowed is False


async def test_prefixes_keep_stores_apart(redis_client: Redis) -> None:
    first, second = BucketStore(redis_client, "one:"), BucketStore(redis_client, "two:")
    await first.take("b", cost=10, capacity=10, now_ms=T0)

    assert (await second.take("b", cost=1, capacity=10, now_ms=T0)).allowed is True


# ---------- refill ----------


async def test_a_bucket_without_refill_never_recovers(store: BucketStore) -> None:
    await take(store, 10)

    later = await take(store, 1, now_ms=T0 + 10 * 86_400_000)  # ten days on

    assert later.allowed is False
    assert later.retry_after is None  # waiting cannot help a fixed budget


async def test_tokens_return_at_the_refill_rate(store: BucketStore) -> None:
    await take(store, 10, refill_per_second=2)

    after_half_a_second = await take(store, 0, refill_per_second=2, now_ms=T0 + 500)
    after_two_seconds = await take(store, 0, refill_per_second=2, now_ms=T0 + 2_000)

    assert after_half_a_second.remaining == 1
    assert after_two_seconds.remaining == 4


async def test_refill_stops_at_capacity(store: BucketStore) -> None:
    await take(store, 10, refill_per_second=2)

    result = await take(store, 0, refill_per_second=2, now_ms=T0 + 3_600_000)

    assert result.remaining == 10


async def test_fractions_of_a_token_accumulate(store: BucketStore) -> None:
    await take(store, 10, refill_per_second=2)  # 0.6 of a token every 300 ms

    steps = [
        (await take(store, 0, refill_per_second=2, now_ms=T0 + 300 * n)).remaining
        for n in range(1, 6)
    ]

    assert steps == [0, 1, 1, 2, 3]  # floor(0.6), floor(1.2), floor(1.8), floor(2.4), floor(3.0)


async def test_retry_after_says_when_the_call_would_succeed(store: BucketStore) -> None:
    await take(store, 7, refill_per_second=4)  # 3 left

    refused = await take(store, 10, refill_per_second=4)  # needs 7 more at 4 a second

    assert refused.allowed is False
    assert refused.retry_after == pytest.approx(1.75)
    ready = await take(store, 10, refill_per_second=4, now_ms=T0 + 1_750)
    assert ready.allowed is True


async def test_time_going_backwards_mints_no_tokens(store: BucketStore) -> None:
    await take(store, 10, refill_per_second=2)

    result = await take(store, 0, refill_per_second=2, now_ms=T0 - 60_000)

    assert result.remaining == 0


async def test_lowering_the_capacity_trims_an_existing_balance(store: BucketStore) -> None:
    await take(store, 0, capacity=100)  # a full bucket of 100
    assert (await take(store, 0, capacity=100)).remaining == 100

    smaller = await take(store, 0, capacity=5)

    assert smaller.remaining == 5


async def test_the_real_clock_refills_too(store: BucketStore) -> None:
    await store.take("rt", cost=5, capacity=5, refill_per_second=20)  # drain, no now_ms

    await asyncio.sleep(0.3)  # about 6 tokens' worth, capped at 5

    result = await store.take("rt", cost=2, capacity=5, refill_per_second=20)
    assert result.allowed is True


# ---------- expiry ----------


async def test_refilling_buckets_expire_once_they_would_be_full_again(
    store: BucketStore, redis_client: Redis
) -> None:
    await store.take("e", cost=1, capacity=10, refill_per_second=5)  # 2 s to refill, plus 1 s

    ttl = await redis_client.pttl("llmrouter:quota:e")

    assert 0 < ttl <= 3_000


async def test_budgets_can_be_given_a_lifetime(store: BucketStore, redis_client: Redis) -> None:
    await store.take("budget", cost=1, capacity=100, ttl_seconds=60)

    ttl = await redis_client.pttl("llmrouter:quota:budget")

    assert 0 < ttl <= 60_000


async def test_budgets_without_a_lifetime_persist(store: BucketStore, redis_client: Redis) -> None:
    await store.take("forever", cost=1, capacity=100)

    assert await redis_client.pttl("llmrouter:quota:forever") == -1


async def test_an_expired_bucket_starts_full_again(store: BucketStore, redis_client: Redis) -> None:
    await store.take("short", cost=10, capacity=10, ttl_seconds=0.1)
    await asyncio.sleep(0.25)

    result = await store.take("short", cost=1, capacity=10, ttl_seconds=0.1)

    assert (result.allowed, result.remaining) == (True, 9)


# ---------- atomicity ----------


async def test_one_hundred_racers_for_ten_tokens_ten_win(store: BucketStore) -> None:
    results = await asyncio.gather(
        *(store.take("race", cost=1, capacity=10, now_ms=T0) for _ in range(100))
    )

    assert sum(r.allowed for r in results) == 10
    assert (await store.take("race", cost=0, capacity=10, now_ms=T0)).remaining == 0


async def test_mixed_costs_never_oversubscribe(store: BucketStore) -> None:
    rng = random.Random(7)
    costs = [rng.randint(1, 9) for _ in range(150)]

    results = await asyncio.gather(
        *(store.take("mixed", cost=c, capacity=100, now_ms=T0) for c in costs)
    )

    spent = sum(c for c, r in zip(costs, results, strict=True) if r.allowed)
    assert spent <= 100
    assert (await store.take("mixed", cost=0, capacity=100, now_ms=T0)).remaining == 100 - spent
    # and it is not needlessly strict: nothing that was refused could have fitted afterwards
    assert all(c > 100 - spent or r.allowed for c, r in zip(costs, results, strict=True))


async def test_many_buckets_race_independently(store: BucketStore) -> None:
    names = [f"user-{n}" for n in range(20)]

    results = await asyncio.gather(
        *(store.take(name, cost=1, capacity=5, now_ms=T0) for name in names for _ in range(30))
    )

    for index, name in enumerate(names):
        wins = sum(r.allowed for r in results[index * 30 : (index + 1) * 30])
        assert wins == 5, name


async def test_a_flushed_script_cache_recovers(store: BucketStore, redis_client: Redis) -> None:
    await take(store, 1)

    await redis_client.script_flush()  # as after a Redis restart

    assert (await take(store, 1)).remaining == 8


# ---------- bad input ----------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"capacity": 0}, "capacity"),
        ({"capacity": -3}, "capacity"),
        ({"capacity": 2**60}, "capacity"),
        ({"cost": -1}, "cost"),
        ({"refill_per_second": -1}, "refill"),
        ({"ttl_seconds": 0}, "ttl"),
        ({"ttl_seconds": -5}, "ttl"),
    ],
)
async def test_invalid_arguments_are_rejected(
    store: BucketStore, kwargs: dict[str, object], message: str
) -> None:
    arguments: dict[str, object] = {"cost": 1, "capacity": 10, **kwargs}

    with pytest.raises(ValueError, match=message):
        await store.take("b", **arguments)  # type: ignore[arg-type]


async def test_a_bucket_needs_a_name(store: BucketStore) -> None:
    with pytest.raises(ValueError, match="name"):
        await store.take("", cost=1, capacity=10)
