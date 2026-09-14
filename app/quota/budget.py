import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from redis.asyncio import Redis

NANO = 1_000_000_000  # money is counted in nano-dollars so tiny per-token prices stay exact
_MAX_EXACT = 2**53  # Lua numbers are doubles


def usd_to_nano(usd: float) -> int:
    return round(usd * NANO)


def nano_to_usd(nano: int) -> float:
    return nano / NANO


def month_period(now: datetime) -> str:
    """The calendar month (UTC) a budget applies to, such as ``2026-11``."""
    return now.astimezone(UTC).strftime("%Y-%m")


def seconds_until_next_month(now: datetime) -> float:
    now = now.astimezone(UTC)
    year, month = (now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1)
    return (datetime(year, month, 1, tzinfo=UTC) - now).total_seconds()


# Spend counters in atomic Redis scripts. Each counter holds how much has been spent against its
# limit. A spend is checked against *every* counter before any is changed, so a request charged
# to both a key and its tenant is either counted on both or on neither, however many gateway
# replicas race. (All of a script's keys must live on one Redis node, as with any multi-key script.)
#
#   KEYS[i]      the counters
#   ARGV[1]      the amount
#   ARGV[1 + i]  the limit of counter i
#   ARGV[1 + n + i]  its expiry in ms (0: none), applied when it has none yet
_SPEND = """
local n = #KEYS
local amount = tonumber(ARGV[1])

local current = {}
for i = 1, n do
  current[i] = tonumber(redis.call('GET', KEYS[i]) or '0')
end

for i = 1, n do
  if current[i] + amount > tonumber(ARGV[1 + i]) then
    local blocked = {0, i}
    for j = 1, n do
      blocked[#blocked + 1] = current[j]
    end
    return blocked
  end
end

local result = {1, 0}
for i = 1, n do
  local spent = redis.call('INCRBY', KEYS[i], string.format('%.0f', amount))
  local ttl = tonumber(ARGV[1 + n + i])
  if ttl > 0 and redis.call('PTTL', KEYS[i]) < 0 then
    redis.call('PEXPIRE', KEYS[i], ttl)
  end
  result[#result + 1] = spent
end
return result
"""

# Add a signed amount to every counter, never letting one go below zero. A charge may push a
# counter past its limit: the money has been spent, so later requests are refused until the
# next period.
_ADJUST = """
local n = #KEYS
local delta = tonumber(ARGV[1])
local result = {}
for i = 1, n do
  local spent = redis.call('INCRBY', KEYS[i], string.format('%.0f', delta))
  if spent < 0 then
    redis.call('SET', KEYS[i], '0', 'KEEPTTL')
    spent = 0
  end
  local ttl = tonumber(ARGV[1 + n + i])
  if ttl > 0 and redis.call('PTTL', KEYS[i]) < 0 then
    redis.call('PEXPIRE', KEYS[i], ttl)
  end
  result[#result + 1] = spent
end
return result
"""


@dataclass(frozen=True)
class Counter:
    name: str
    limit: int  # in nano-dollars
    ttl_seconds: float = 0  # 0: the counter never expires


@dataclass(frozen=True)
class SpendResult:
    allowed: bool
    blocked_by: str | None  # the counter that refused, when not allowed
    spent: dict[str, int]  # every counter's total (before the attempt, if it was refused)


class BudgetStore:
    def __init__(self, redis: Redis, prefix: str = "llmrouter:budget:") -> None:
        self._prefix = prefix
        self._redis = redis
        self._spend = redis.register_script(_SPEND)
        self._adjust = redis.register_script(_ADJUST)

    def _prepare(
        self, counters: Sequence[Counter], amount: int, *, signed: bool
    ) -> tuple[list[str], list[int | float | str]]:
        if not counters:
            raise ValueError("at least one counter is required")
        low = -_MAX_EXACT if signed else 0
        if not low <= amount <= _MAX_EXACT:
            raise ValueError(f"amount must be between {low} and {_MAX_EXACT}")
        for counter in counters:
            if not counter.name:
                raise ValueError("a counter needs a name")
            if not 0 <= counter.limit <= _MAX_EXACT:
                raise ValueError(f"limit must be between 0 and {_MAX_EXACT}")
            if counter.ttl_seconds < 0:
                raise ValueError("ttl_seconds must not be negative")
        keys = [self._prefix + counter.name for counter in counters]
        args: list[int | float | str] = [amount]
        args += [counter.limit for counter in counters]
        args += [int(counter.ttl_seconds * 1000) for counter in counters]
        return keys, args

    async def spend(self, counters: Sequence[Counter], amount: int) -> SpendResult:
        """Add ``amount`` to every counter, or to none if any would pass its limit."""
        keys, args = self._prepare(counters, amount, signed=False)
        raw = await self._spend(keys=keys, args=args)
        allowed, blocked_index, *totals = raw
        spent = {counter.name: int(total) for counter, total in zip(counters, totals, strict=True)}
        blocked = None if allowed else counters[int(blocked_index) - 1].name
        return SpendResult(allowed=bool(allowed), blocked_by=blocked, spent=spent)

    async def adjust(self, counters: Sequence[Counter], delta: int) -> dict[str, int]:
        """Add ``delta`` (negative to refund) to every counter; returns the new totals."""
        keys, args = self._prepare(counters, delta, signed=True)
        totals = await self._adjust(keys=keys, args=args)
        return {c.name: int(t) for c, t in zip(counters, totals, strict=True)}

    async def spent(self, name: str) -> int:
        raw = await self._redis.get(self._prefix + name)
        return 0 if raw is None else int(raw)


def cost_nano(usd: float) -> int:
    """Round a cost up to whole nano-dollars (the rounding always favours the gateway)."""
    return math.ceil(round(usd * NANO, 6))
