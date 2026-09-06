from redis.asyncio import BlockingConnectionPool, Redis

from app.core.config import Settings


def build_redis_client(settings: Settings) -> Redis:
    """An async Redis client. Connections are made lazily, so building one never blocks startup.

    Two things matter because quota checks sit on the request path:

    * Short timeouts, so a hung Redis becomes a quick, handled failure instead of stalled requests.
    * A *blocking* pool. The default pool raises "Too many connections" the instant a burst of
      requests exceeds its size, which would look like Redis being down. This one makes the extra
      callers wait for a free connection, for at most ``redis_timeout`` seconds.
    """
    pool = BlockingConnectionPool.from_url(
        settings.redis_url,
        max_connections=settings.redis_max_connections,
        timeout=settings.redis_timeout,
        decode_responses=True,
        socket_connect_timeout=settings.redis_timeout,
        socket_timeout=settings.redis_timeout,
        health_check_interval=30,
    )
    client = Redis(connection_pool=pool)
    # redis-py forces this to False for a pool you supply (whatever the keyword says), which
    # would make aclose() leave every connection open. We own the pool, so close it with the client.
    client.auto_close_connection_pool = True
    return client


async def redis_ready(client: Redis) -> bool:
    try:
        return bool(await client.ping())
    except Exception:
        return False
