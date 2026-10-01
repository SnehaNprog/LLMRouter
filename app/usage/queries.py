from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import ColumnElement, String, and_, case, cast, delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import Label
from sqlalchemy.sql.expression import SQLColumnExpression

from app.db.models import RequestLog
from app.schemas.usage import NANO, RequestLogOut, UsageFigures, UsageGroup, UsageReport

GroupBy = Literal["none", "model", "requested_model", "key", "day", "tier", "status"]
MAX_GROUPS = 500


def month_start(now: datetime | None = None) -> datetime:
    """The first instant of the current calendar month (UTC), the period budgets run on."""
    now = (now or datetime.now(UTC)).astimezone(UTC)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def as_utc(value: datetime) -> datetime:
    """A naive datetime from a query string is taken to be UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _group_expression(group_by: GroupBy) -> SQLColumnExpression[str]:
    if group_by == "model":
        return func.coalesce(RequestLog.served_model, "(none)")
    if group_by == "requested_model":
        return RequestLog.requested_model
    if group_by == "key":
        return RequestLog.key_id
    if group_by == "day":
        return RequestLog.day
    if group_by == "tier":
        return func.coalesce(RequestLog.tier, "(none)")
    if group_by == "status":
        return cast(RequestLog.status_code, String)
    raise ValueError(f"cannot group by '{group_by}'")


def _filters(
    tenant_id: str, start: datetime, end: datetime | None, key_id: str | None
) -> list[ColumnElement[bool]]:
    conditions = [RequestLog.tenant_id == tenant_id, RequestLog.created_at >= as_utc(start)]
    if end is not None:
        conditions.append(RequestLog.created_at < as_utc(end))
    if key_id is not None:
        conditions.append(RequestLog.key_id == key_id)
    return conditions


def _figures(row: tuple[object, ...]) -> UsageFigures:
    requests, errors, prompt, completion, cost, latency = row
    prompt_i, completion_i = int(prompt or 0), int(completion or 0)  # type: ignore[call-overload]
    return UsageFigures(
        requests=int(requests or 0),  # type: ignore[call-overload]
        errors=int(errors or 0),  # type: ignore[call-overload]
        prompt_tokens=prompt_i,
        completion_tokens=completion_i,
        total_tokens=prompt_i + completion_i,
        cost_usd=int(cost or 0) / NANO,  # type: ignore[call-overload]
        avg_latency_ms=round(float(latency or 0.0), 1),  # type: ignore[arg-type]
    )


_AGGREGATES = (
    func.count(),
    func.sum(
        case(
            (or_(RequestLog.status_code >= 400, RequestLog.error_code.is_not(None)), 1),
            else_=0,
        )
    ),
    func.sum(RequestLog.prompt_tokens),
    func.sum(RequestLog.completion_tokens),
    func.sum(RequestLog.cost_nano_usd),
    func.avg(RequestLog.latency_ms),
)


async def usage_report(
    session: AsyncSession,
    tenant_id: str,
    *,
    start: datetime,
    end: datetime | None = None,
    group_by: GroupBy = "model",
    key_id: str | None = None,
) -> UsageReport:
    """What a tenant (optionally one of its keys) used between ``start`` and ``end``.

    Cost is the sum of what each request cost when it was made, so a later price change in the
    config does not rewrite history.
    """
    where = _filters(tenant_id, start, end, key_id)
    totals_row = (await session.execute(select(*_AGGREGATES).where(*where))).one()
    groups: list[UsageGroup] = []
    if group_by != "none":
        key: Label[str] = _group_expression(group_by).label("key")
        order = key.asc() if group_by == "day" else func.count().desc()
        rows = await session.execute(
            select(key, *_AGGREGATES)
            .where(*where)
            .group_by(key)
            .order_by(order, key.asc())
            .limit(MAX_GROUPS)
        )
        groups = [
            UsageGroup(key=str(row[0]), **_figures(tuple(row[1:])).model_dump()) for row in rows
        ]
    return UsageReport(
        start=as_utc(start),
        end=None if end is None else as_utc(end),
        group_by=group_by,
        totals=_figures(tuple(totals_row)),
        groups=groups,
    )


class BadCursorError(ValueError):
    """A ``before`` cursor that this service did not issue."""


def encode_cursor(row: RequestLogOut) -> str:
    return f"{row.created_at.isoformat()}~{row.id}"


def decode_cursor(cursor: str) -> tuple[datetime, str]:
    stamp, separator, row_id = cursor.partition("~")
    if not separator or not row_id:
        raise BadCursorError("before is not a cursor from a previous page")
    try:
        return as_utc(datetime.fromisoformat(stamp)), row_id
    except ValueError:
        raise BadCursorError("before is not a cursor from a previous page") from None


async def list_requests(
    session: AsyncSession,
    tenant_id: str,
    *,
    start: datetime | None = None,
    before: str | None = None,
    key_id: str | None = None,
    model: str | None = None,
    only_errors: bool = False,
    limit: int = 50,
) -> tuple[list[RequestLogOut], str | None]:
    """The tenant's most recent requests, newest first, and the cursor for the next page.

    Pages are cut at a (time, id) position, so rows that share a timestamp are neither skipped
    nor repeated across a page boundary.
    """
    query = select(RequestLog).where(RequestLog.tenant_id == tenant_id)
    if start is not None:
        query = query.where(RequestLog.created_at >= as_utc(start))
    if before is not None:
        stamp, row_id = decode_cursor(before)
        query = query.where(
            or_(
                RequestLog.created_at < stamp,
                and_(RequestLog.created_at == stamp, RequestLog.id > row_id),
            )
        )
    if key_id is not None:
        query = query.where(RequestLog.key_id == key_id)
    if model is not None:
        query = query.where(
            or_(RequestLog.served_model == model, RequestLog.requested_model == model)
        )
    if only_errors:
        query = query.where(or_(RequestLog.status_code >= 400, RequestLog.error_code.is_not(None)))
    rows = (
        (
            await session.execute(
                query.order_by(RequestLog.created_at.desc(), RequestLog.id).limit(limit + 1)
            )
        )
        .scalars()
        .all()
    )
    page = rows[:limit]
    more = len(rows) > limit
    out = [RequestLogOut.of(r) for r in page]
    return out, (encode_cursor(out[-1]) if more and out else None)


async def prune_request_logs(session: AsyncSession, older_than: datetime) -> int:
    """Delete log rows from before ``older_than``; returns how many went."""
    result = await session.execute(
        delete(RequestLog).where(RequestLog.created_at < as_utc(older_than))
    )
    await session.commit()
    return int(result.rowcount or 0)  # type: ignore[attr-defined]
