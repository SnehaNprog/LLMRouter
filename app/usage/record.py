import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.logging import request_id_var
from app.db.models import RequestLog
from app.quota.budget import cost_nano
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

    def to_row(self, now: datetime | None = None) -> RequestLog:
        now = now or datetime.now(UTC)
        deployment = self.served.deployment if self.served is not None else None
        price = (
            deployment.cost(self.prompt_tokens, self.completion_tokens)
            if deployment is not None
            else 0.0
        )
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
            latency_ms=round((time.perf_counter() - self.started) * 1000),
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

    async def write(self, entry: LogEntry) -> None:
        if not self.enabled or self._factory is None:
            return
        try:
            async with self._factory() as session:
                session.add(entry.to_row())
                await session.commit()
        except Exception:
            logger.exception("could not write the request log", extra={"tenant": entry.tenant_id})
