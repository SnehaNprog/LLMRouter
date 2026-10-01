from datetime import datetime

from pydantic import BaseModel

from app.db.models import RequestLog

NANO = 1_000_000_000


class UsageFigures(BaseModel):
    requests: int
    errors: int  # status 400 or above, or a stream that broke after it had begun
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    avg_latency_ms: float


class UsageGroup(UsageFigures):
    key: str  # the model, key, day... this row is about


class UsageReport(BaseModel):
    start: datetime
    end: datetime | None  # None: up to now
    group_by: str
    totals: UsageFigures
    groups: list[UsageGroup]


class RequestLogOut(BaseModel):
    id: str
    created_at: datetime
    request_id: str | None
    key_id: str
    requested_model: str
    served_model: str | None
    status_code: int
    error_code: str | None
    stream: bool
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: int
    routed: bool
    tier: str | None
    escalated: bool
    complexity: float | None

    @classmethod
    def of(cls, row: RequestLog) -> "RequestLogOut":
        return cls(
            id=row.id,
            created_at=row.created_at,
            request_id=row.request_id,
            key_id=row.key_id,
            requested_model=row.requested_model,
            served_model=row.served_model,
            status_code=row.status_code,
            error_code=row.error_code,
            stream=row.stream,
            prompt_tokens=row.prompt_tokens,
            completion_tokens=row.completion_tokens,
            cost_usd=row.cost_nano_usd / NANO,
            latency_ms=row.latency_ms,
            routed=row.routed,
            tier=row.tier,
            escalated=row.escalated,
            complexity=row.complexity,
        )


class RequestLogList(BaseModel):
    data: list[RequestLogOut]
    next_before: str | None  # pass as ?before= to get the next, older page
