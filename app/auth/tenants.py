from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.service import INHERIT, TenantNotFoundError, _Inherit, budget_to_nano
from app.db.models import Tenant


class AdminConflictError(Exception):
    """The request is valid but clashes with the current state (HTTP 409)."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class TenantExistsError(AdminConflictError):
    def __init__(self, name: str) -> None:
        super().__init__(f"A tenant named '{name}' already exists.", "tenant_exists")


def clean_tenant_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("a tenant needs a name")
    if len(name) > 200:
        raise ValueError("tenant names are limited to 200 characters")
    return name


async def create_tenant(
    session: AsyncSession, name: str, monthly_budget_usd: float | None = None
) -> Tenant:
    tenant = Tenant(
        name=clean_tenant_name(name), monthly_budget_nano_usd=budget_to_nano(monthly_budget_usd)
    )
    session.add(tenant)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise TenantExistsError(tenant.name) from None
    return tenant


async def get_tenant(session: AsyncSession, tenant_id: str) -> Tenant:
    tenant = await session.get(Tenant, tenant_id)
    if tenant is None:
        raise TenantNotFoundError(tenant_id)
    return tenant


async def list_tenants(session: AsyncSession, *, limit: int = 50, offset: int = 0) -> list[Tenant]:
    result = await session.execute(
        select(Tenant).order_by(Tenant.created_at, Tenant.id).limit(limit).offset(offset)
    )
    return list(result.scalars())


async def update_tenant(
    session: AsyncSession,
    tenant_id: str,
    *,
    name: str | None = None,
    is_active: bool | None = None,
    monthly_budget_usd: float | None | _Inherit = INHERIT,
) -> Tenant:
    """Change a tenant. ``monthly_budget_usd=None`` removes the budget; leaving it out keeps it."""
    if name is None and is_active is None and isinstance(monthly_budget_usd, _Inherit):
        raise ValueError("nothing to update: send a name, is_active and/or monthly_budget_usd")
    budget_nano = (
        INHERIT if isinstance(monthly_budget_usd, _Inherit) else budget_to_nano(monthly_budget_usd)
    )
    tenant = await get_tenant(session, tenant_id)
    if name is not None:
        tenant.name = clean_tenant_name(name)
    if is_active is not None:
        tenant.is_active = is_active
    if not isinstance(budget_nano, _Inherit):
        tenant.monthly_budget_nano_usd = budget_nano
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise TenantExistsError(name or tenant.name) from None
    return tenant


async def delete_tenant(session: AsyncSession, tenant_id: str) -> None:
    """Delete a tenant and, through the database cascade, all of its keys."""
    tenant = await get_tenant(session, tenant_id)
    await session.delete(tenant)
    await session.commit()
