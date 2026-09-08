from dataclasses import dataclass

from redis.asyncio import Redis

# A token bucket in one atomic Redis script. Reading the balance, refilling it for the time
# that has passed, checking it and taking from it all happen inside a single script run, so
# concurrent callers (across any number of gateway replicas) can never oversubscribe it.
#
#   KEYS[1]  the bucket (a hash holding the balance and when it was last updated)
#   ARGV[1]  capacity, the most the bucket can hold
#   ARGV[2]  refill rate in tokens per second (0: the bucket never refills)
#   ARGV[3]  cost of this call
#   ARGV[4]  expiry in ms for the bucket (0: derive it from the refill time, if there is one)
#   ARGV[5]  "now" in ms, or "" to use Redis's own clock (the override exists for tests)
#
# Returns {allowed (0/1), tokens left, retry_after_ms}; retry_after_ms is -1 when waiting
# can never help (the cost exceeds the capacity, or the bucket does not refill).
_TAKE = """
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])

local now
if ARGV[5] ~= '' then
  now = tonumber(ARGV[5])
else
  local t = redis.call('TIME')
  now = t[1] * 1000 + math.floor(t[2] / 1000)
end

local state = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])
if tokens == nil or ts == nil then
  tokens = capacity
  ts = now
end

local elapsed = now - ts
if elapsed > 0 and rate > 0 then
  tokens = tokens + elapsed * rate / 1000
end
if tokens > capacity then
  tokens = capacity
end
if now > ts then
  ts = now
end

local allowed = 0
local retry_ms = 0
if cost > capacity then
  retry_ms = -1
elseif tokens >= cost then
  allowed = 1
  tokens = tokens - cost
elseif rate > 0 then
  retry_ms = math.ceil((cost - tokens) / rate * 1000)
else
  retry_ms = -1
end

redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', string.format('%.0f', ts))
if ttl > 0 then
  redis.call('PEXPIRE', KEYS[1], ttl)
elseif rate > 0 then
  redis.call('PEXPIRE', KEYS[1], math.ceil(capacity / rate * 1000) + 1000)
end

return {allowed, math.floor(tokens), retry_ms}
"""

_MAX_EXACT = 2**53  # Lua numbers are doubles


@dataclass(frozen=True)
class TakeResult:
    allowed: bool
    remaining: int  # whole tokens left after this call
    retry_after: float | None  # seconds until the same call would succeed; None: never


class BucketStore:
    """Token buckets stored in Redis. Used for request, token and budget quotas."""

    def __init__(self, redis: Redis, prefix: str = "llmrouter:quota:") -> None:
        self._redis = redis
        self._prefix = prefix
        self._take = redis.register_script(_TAKE)

    async def take(
        self,
        name: str,
        *,
        cost: int,
        capacity: int,
        refill_per_second: float = 0,
        ttl_seconds: float | None = None,
        now_ms: int | None = None,
    ) -> TakeResult:
        """Atomically take ``cost`` tokens from the bucket ``name`` if it has enough.

        A new bucket starts full. With ``refill_per_second`` the balance climbs back towards
        ``capacity`` as time passes; without it the bucket is a fixed budget. A cost of 0 reads
        the balance without spending anything. ``ttl_seconds`` expires the bucket (a budget for
        one billing period, say); rate-limited buckets expire on their own once they would have
        refilled completely. ``now_ms`` replaces the Redis clock, for tests.
        """
        if not name:
            raise ValueError("a bucket needs a name")
        if not 1 <= capacity <= _MAX_EXACT:
            raise ValueError(f"capacity must be between 1 and {_MAX_EXACT}")
        if not 0 <= cost <= _MAX_EXACT:
            raise ValueError(f"cost must be between 0 and {_MAX_EXACT}")
        if refill_per_second < 0:
            raise ValueError("refill_per_second must not be negative")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")

        allowed, remaining, retry_ms = await self._take(
            keys=[self._prefix + name],
            args=[
                capacity,
                refill_per_second,
                cost,
                0 if ttl_seconds is None else int(ttl_seconds * 1000),
                "" if now_ms is None else now_ms,
            ],
        )
        retry_after = None if retry_ms < 0 else retry_ms / 1000
        return TakeResult(allowed=bool(allowed), remaining=int(remaining), retry_after=retry_after)
