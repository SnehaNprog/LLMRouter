import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.logging import request_id_var
from app.db.models import RequestLog
from app.quota.budget import cost_nano
from app.router.deployment import Deployment
from app.router.router import Served
from app.routing.selector import RoutingChoice

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """A request being logged: filled in as it moves through the gateway, written once it ends."""

    tenant_id: str
    key_id: str
    requested_model: str
    stream: bool
    request_id: str | None = field(default_factory=request_id_var.get)
    started: float = field(default_factory=time.perf_counter)
    status_code: int = 200
    error_code: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    served: Served | None = None  # filled in by the router with the deployment that answered
    choice: RoutingChoice | None = None  # set when the request asked for "auto"

    def succeed(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens, self.completion_tokens = prompt_tokens, completion_tokens

    def fail(self, status_code: int, error_code: str | None) -> None:
        self.status_code, self.error_code = status_code, error_code

    @property
    def deployment(self) -> Deployment | None:
        return self.served.deployment if self.served is not None else None

    def price_usd(self) -> float:
        """What the request cost, at the answering deployment's rates (0 if none answered)."""
        deployment = self.deployment
        if deployment is None:
            return 0.0
        return deployment.cost(self.prompt_tokens, self.completion_tokens)

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def to_row(self, now: datetime | None = None) -> RequestLog:
        now = now or datetime.now(UTC)
        deployment = self.deployment
        price = self.price_usd()
        return RequestLog(
            created_at=now,
            day=now.astimezone(UTC).strftime("%Y-%m-%d"),
            request_id=self.request_id,
            tenant_id=self.tenant_id,
            key_id=self.key_id,
            requested_model=self.requested_model,
            served_model=deployment.alias if deployment is not None else None,
            deployment_id=deployment.id if deployment is not None else None,
            status_code=self.status_code,
            error_code=self.error_code,
            stream=self.stream,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            cost_nano_usd=cost_nano(price),
            latency_ms=round(self.elapsed() * 1000),
            routed=self.choice is not None,
            tier=(
                deployment.config.tier.value
                if deployment is not None and deployment.config.tier is not None
                else None
            ),
            escalated=(
                self.choice is not None
                and deployment is not None
                and deployment.alias != self.choice.alias
            ),
            complexity=None if self.choice is None else round(self.choice.complexity.value, 4),
        )


class RequestLogger:
    """Writes request logs. A failure to log is logged and swallowed: it must never fail, or
    slow with a retry, the request that was being served."""

    def __init__(
        self, factory: async_sessionmaker[AsyncSession] | None, *, enabled: bool = True
    ) -> None:
        self._factory = factory
        self.enabled = enabled and factory is not None
        self._sinks: list[Callable[[LogEntry], None]] = []

    def add_sink(self, sink: Callable[[LogEntry], None]) -> None:
        """Also hand every finished request to ``sink`` (metrics, say), logging enabled or not."""
        self._sinks.append(sink)

    async def write(self, entry: LogEntry) -> None:
        for sink in self._sinks:
            try:
                sink(entry)
            except Exception:  # an observer's bug must never fail the request it observes
                logger.exception("a request observer failed")
        if not self.enabled or self._factory is None:
            return
        try:
            async with self._factory() as session:
                session.add(entry.to_row())
                await session.commit()
        except Exception:
            logger.exception("could not write the request log", extra={"tenant": entry.tenant_id})
