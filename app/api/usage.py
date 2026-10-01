from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import AuthContext, require_scope
from app.auth.scopes import ADMIN_TENANTS, USAGE_READ
from app.auth.tenants import get_tenant
from app.db.session import get_session
from app.schemas.usage import RequestLogList, UsageReport
from app.usage.queries import (
    BadCursorError,
    GroupBy,
    as_utc,
    list_requests,
    month_start,
    usage_report,
)

Session = Annotated[AsyncSession, Depends(get_session)]

own = APIRouter(prefix="/v1/usage", tags=["usage"])
admin = APIRouter(prefix="/admin/tenants/{tenant_id}/usage", tags=["admin"])

Start = Annotated[
    datetime | None,
    Query(description="From this instant (ISO 8601; no zone means UTC). Default: this month."),
]
End = Annotated[datetime | None, Query(description="Before this instant. Default: up to now.")]
Group = Annotated[GroupBy, Query(description="How to break the figures down.")]
KeyFilter = Annotated[str | None, Query(description="Only this API key's requests.")]
Limit = Annotated[int, Query(ge=1, le=200)]


def _window(start: datetime | None, end: datetime | None) -> tuple[datetime, datetime | None]:
    begin = as_utc(start) if start is not None else month_start()
    until = as_utc(end) if end is not None else None
    if until is not None and until <= begin:
        raise HTTPException(status_code=400, detail="end must be after start")
    return begin, until


async def _report(
    session: AsyncSession,
    tenant_id: str,
    start: datetime | None,
    end: datetime | None,
    group_by: GroupBy,
    key_id: str | None,
) -> UsageReport:
    begin, until = _window(start, end)
    return await usage_report(
        session, tenant_id, start=begin, end=until, group_by=group_by, key_id=key_id
    )


async def _requests(
    session: AsyncSession,
    tenant_id: str,
    start: datetime | None,
    before: str | None,
    key_id: str | None,
    model: str | None,
    errors: bool,
    limit: int,
) -> RequestLogList:
    try:
        data, cursor = await list_requests(
            session,
            tenant_id,
            start=None if start is None else as_utc(start),
            before=before,
            key_id=key_id,
            model=model,
            only_errors=errors,
            limit=limit,
        )
    except BadCursorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return RequestLogList(data=data, next_before=cursor)


# ---- a tenant's own usage (any key with usage:read) ----


@own.get("")
async def my_usage(
    session: Session,
    auth: Annotated[AuthContext, Depends(require_scope(USAGE_READ))],
    start: Start = None,
    end: End = None,
    group_by: Group = "model",
    key_id: KeyFilter = None,
) -> UsageReport:
    """What this organisation has used: requests, tokens and cost, with a breakdown.

    Only the caller's own organisation is ever visible, whatever ``key_id`` is asked for.
    """
    return await _report(session, auth.tenant_id, start, end, group_by, key_id)


@own.get("/requests")
async def my_requests(
    session: Session,
    auth: Annotated[AuthContext, Depends(require_scope(USAGE_READ))],
    start: Start = None,
    before: Annotated[str | None, Query(description="Page cursor from next_before.")] = None,
    key_id: KeyFilter = None,
    model: Annotated[str | None, Query(description="Only this model.")] = None,
    errors: Annotated[bool, Query(description="Only failed requests.")] = False,
    limit: Limit = 50,
) -> RequestLogList:
    """This organisation's most recent requests, newest first. No prompt or reply is kept."""
    return await _requests(session, auth.tenant_id, start, before, key_id, model, errors, limit)


# ---- any tenant's usage (administrators) ----


@admin.get("")
async def tenant_usage(
    tenant_id: str,
    session: Session,
    _: Annotated[AuthContext, Depends(require_scope(ADMIN_TENANTS))],
    start: Start = None,
    end: End = None,
    group_by: Group = "model",
    key_id: KeyFilter = None,
) -> UsageReport:
    await get_tenant(session, tenant_id)  # a 404 for a tenant that does not exist
    return await _report(session, tenant_id, start, end, group_by, key_id)


@admin.get("/requests")
async def tenant_requests(
    tenant_id: str,
    session: Session,
    _: Annotated[AuthContext, Depends(require_scope(ADMIN_TENANTS))],
    start: Start = None,
    before: Annotated[str | None, Query(description="Page cursor from next_before.")] = None,
    key_id: KeyFilter = None,
    model: Annotated[str | None, Query(description="Only this model.")] = None,
    errors: Annotated[bool, Query(description="Only failed requests.")] = False,
    limit: Limit = 50,
) -> RequestLogList:
    await get_tenant(session, tenant_id)
    return await _requests(session, tenant_id, start, before, key_id, model, errors, limit)
