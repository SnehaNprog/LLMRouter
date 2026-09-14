import asyncio
import random
from datetime import UTC, datetime, timedelta, timezone

import pytest
from redis.asyncio import Redis

from app.quota.budget import (
    BudgetStore,
    Counter,
    cost_nano,
    month_period,
    nano_to_usd,
    seconds_until_next_month,
    usd_to_nano,
)


@pytest.fixture
def store(redis_client: Redis) -> BudgetStore:
    return BudgetStore(redis_client)


def counter(name: str = "a", limit: int = 100, ttl: float = 0) -> Counter:
    return Counter(name, limit, ttl)


# ---------- a single counter ----------


async def test_spending_within_the_limit_is_allowed(store: BudgetStore) -> None:
    first = await store.spend([counter()], 30)
    second = await store.spend([counter()], 50)

    assert (first.allowed, first.spent) == (True, {"a": 30})
    assert (second.allowed, second.spent) == (True, {"a": 80})


async def test_the_limit_can_be_reached_exactly(store: BudgetStore) -> None:
    assert (await store.spend([counter()], 100)).allowed is True
    assert (await store.spend([counter()], 1)).allowed is False


async def test_a_refused_spend_changes_nothing(store: BudgetStore) -> None:
    await store.spend([counter()], 90)

    refused = await store.spend([counter()], 20)

    assert (refused.allowed, refused.blocked_by, refused.spent) == (False, "a", {"a": 90})
    assert await store.spent("a") == 90
    assert (await store.spend([counter()], 10)).allowed is True


async def test_spending_nothing_is_always_fine(store: BudgetStore) -> None:
    await store.spend([counter()], 100)

    assert (await store.spend([counter()], 0)).allowed is True


async def test_counters_are_independent(store: BudgetStore) -> None:
    await store.spend([counter("a")], 100)

    assert (await store.spend([counter("b")], 100)).allowed is True


async def test_the_limit_is_read_on_every_call(store: BudgetStore) -> None:
    await store.spend([counter(limit=100)], 100)
    assert (await store.spend([counter(limit=100)], 1)).allowed is False

    raised = await store.spend([counter(limit=150)], 40)  # the budget was increased mid-period
    lowered = await store.spend([counter(limit=50)], 1)  # and then cut below what is spent

    assert raised.allowed is True
    assert lowered.allowed is False


async def test_a_limit_of_zero_blocks_everything(store: BudgetStore) -> None:
    assert (await store.spend([counter(limit=0)], 1)).allowed is False


# ---------- several counters, all or nothing ----------


async def test_a_spend_is_charged_to_every_counter(store: BudgetStore) -> None:
    result = await store.spend([counter("key", 100), counter("tenant", 1000)], 40)

    assert result.allowed is True
    assert result.spent == {"key": 40, "tenant": 40}


async def test_one_counter_refusing_blocks_them_all(store: BudgetStore) -> None:
    both = [counter("key", 100), counter("tenant", 50)]

    result = await store.spend(both, 60)

    assert (result.allowed, result.blocked_by) == (False, "tenant")
    assert await store.spent("key") == 0  # the key was not charged either
    assert await store.spent("tenant") == 0


async def test_the_first_refusing_counter_is_reported(store: BudgetStore) -> None:
    result = await store.spend([counter("a", 10), counter("b", 10)], 20)

    assert result.blocked_by == "a"


async def test_racing_across_two_counters_never_oversubscribes_either(store: BudgetStore) -> None:
    both = [counter("key", 10), counter("tenant", 6)]

    results = await asyncio.gather(*(store.spend(both, 1) for _ in range(100)))

    assert sum(r.allowed for r in results) == 6
    assert await store.spent("key") == 6
    assert await store.spent("tenant") == 6  # still in step


async def test_a_hundred_racers_for_ten_units_ten_win(store: BudgetStore) -> None:
    results = await asyncio.gather(*(store.spend([counter(limit=10)], 1) for _ in range(100)))

    assert sum(r.allowed for r in results) == 10
    assert await store.spent("a") == 10


async def test_mixed_amounts_never_exceed_the_limit(store: BudgetStore) -> None:
    rng = random.Random(11)
    amounts = [rng.randint(1, 9) for _ in range(150)]

    results = await asyncio.gather(*(store.spend([counter(limit=100)], a) for a in amounts))

    spent = sum(a for a, r in zip(amounts, results, strict=True) if r.allowed)
    assert spent <= 100
    assert await store.spent("a") == spent


# ---------- adjusting ----------


async def test_a_negative_adjustment_refunds(store: BudgetStore) -> None:
    await store.spend([counter()], 70)

    totals = await store.adjust([counter()], -30)

    assert totals == {"a": 40}


async def test_a_positive_adjustment_charges_even_past_the_limit(store: BudgetStore) -> None:
    await store.spend([counter()], 90)

    totals = await store.adjust([counter()], 50)

    assert totals == {"a": 140}
    assert (await store.spend([counter()], 1)).allowed is False  # over budget until the period ends


async def test_a_refund_cannot_go_below_zero(store: BudgetStore) -> None:
    await store.spend([counter()], 10)

    assert await store.adjust([counter()], -500) == {"a": 0}
    assert await store.spent("a") == 0


async def test_adjusting_applies_to_every_counter(store: BudgetStore) -> None:
    both = [counter("key"), counter("tenant")]
    await store.spend(both, 60)

    assert await store.adjust(both, -25) == {"key": 35, "tenant": 35}


async def test_concurrent_adjustments_lose_nothing(store: BudgetStore) -> None:
    await store.spend([counter(limit=10_000)], 5_000)

    await asyncio.gather(
        *(store.adjust([counter(limit=10_000)], 2) for _ in range(200)),
        *(store.adjust([counter(limit=10_000)], -1) for _ in range(100)),
    )

    assert await store.spent("a") == 5_000 + 400 - 100


# ---------- expiry ----------


async def test_counters_expire_with_their_period(store: BudgetStore, redis_client: Redis) -> None:
    await store.spend([counter(ttl=60)], 5)

    assert 0 < await redis_client.pttl("llmrouter:budget:a") <= 60_000


async def test_the_expiry_is_not_pushed_back_by_later_spends(
    store: BudgetStore, redis_client: Redis
) -> None:
    await store.spend([counter(ttl=60)], 1)
    await redis_client.pexpire("llmrouter:budget:a", 5_000)

    await store.spend([counter(ttl=60)], 1)
    await store.adjust([counter(ttl=60)], 1)

    assert await redis_client.pttl("llmrouter:budget:a") <= 5_000  # the period's end stays put


async def test_a_refund_to_zero_keeps_the_expiry(store: BudgetStore, redis_client: Redis) -> None:
    await store.spend([counter(ttl=60)], 5)

    await store.adjust([counter(ttl=60)], -999)

    assert 0 < await redis_client.pttl("llmrouter:budget:a") <= 60_000


async def test_counters_without_a_ttl_persist(store: BudgetStore, redis_client: Redis) -> None:
    await store.spend([counter()], 5)

    assert await redis_client.pttl("llmrouter:budget:a") == -1


async def test_an_expired_counter_starts_afresh(store: BudgetStore) -> None:
    await store.spend([counter(ttl=0.1)], 100)
    await asyncio.sleep(0.25)

    assert (await store.spend([counter(ttl=0.1)], 100)).allowed is True


async def test_a_flushed_script_cache_recovers(store: BudgetStore, redis_client: Redis) -> None:
    await store.spend([counter()], 1)

    await redis_client.script_flush()

    assert (await store.spend([counter()], 1)).spent == {"a": 2}


async def test_prefixes_keep_stores_apart(redis_client: Redis) -> None:
    one, two = BudgetStore(redis_client, "one:"), BudgetStore(redis_client, "two:")
    await one.spend([counter()], 100)

    assert (await two.spend([counter()], 100)).allowed is True


# ---------- bad input ----------


@pytest.mark.parametrize(
    ("counters", "amount", "message"),
    [
        ([], 1, "at least one counter"),
        ([Counter("", 10)], 1, "name"),
        ([Counter("a", -1)], 1, "limit"),
        ([Counter("a", 2**60)], 1, "limit"),
        ([Counter("a", 10, -1)], 1, "ttl"),
        ([Counter("a", 10)], -1, "amount"),
        ([Counter("a", 10)], 2**60, "amount"),
    ],
)
async def test_invalid_spends_are_rejected(
    store: BudgetStore, counters: list[Counter], amount: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        await store.spend(counters, amount)


async def test_invalid_adjustments_are_rejected(store: BudgetStore) -> None:
    with pytest.raises(ValueError, match="amount"):
        await store.adjust([counter()], 2**60)
    with pytest.raises(ValueError, match="at least one"):
        await store.adjust([], 1)


# ---------- money and months ----------


def test_dollars_convert_to_exact_nano_dollars() -> None:
    assert usd_to_nano(1) == 1_000_000_000
    assert usd_to_nano(0.1) == 100_000_000  # no floating-point dust
    assert usd_to_nano(12.345678) == 12_345_678_000
    assert nano_to_usd(2_500_000_000) == 2.5


def test_costs_round_up_never_down() -> None:
    assert cost_nano(0.0) == 0
    assert cost_nano(0.0000000004) == 1  # a fraction of a nano-dollar still counts
    assert cost_nano(0.002) == 2_000_000


def test_the_period_is_the_utc_month() -> None:
    assert month_period(datetime(2026, 11, 2, tzinfo=UTC)) == "2026-11"
    assert month_period(datetime(2026, 12, 31, 23, 59, tzinfo=UTC)) == "2026-12"
    late_in_new_york = datetime(2026, 11, 30, 22, 0, tzinfo=timezone(timedelta(hours=-5)))
    assert month_period(late_in_new_york) == "2026-12"  # already December in UTC


def test_seconds_until_the_next_month() -> None:
    assert seconds_until_next_month(datetime(2026, 11, 30, 23, 0, tzinfo=UTC)) == 3_600
    assert seconds_until_next_month(datetime(2026, 12, 31, 0, 0, tzinfo=UTC)) == 86_400  # new year
    assert seconds_until_next_month(datetime(2026, 2, 1, tzinfo=UTC)) == 28 * 86_400
