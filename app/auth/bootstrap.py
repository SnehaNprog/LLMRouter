from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.scopes import ADMIN_KEYS, ADMIN_TENANTS
from app.auth.service import create_api_key
from app.auth.tenants import clean_tenant_name
from app.db.models import ApiKey, Tenant


async def bootstrap_admin(
    session: AsyncSession, tenant_name: str = "platform", key_name: str = "bootstrap admin"
) -> tuple[Tenant, ApiKey, str]:
    """Create (or reuse) an administrators' tenant and mint an admin key for it.

    The admin API needs an admin key, and only the admin API creates keys, so the first one
    has to come from here. It can administer tenants and keys but cannot call models.
    """
    tenant_name = clean_tenant_name(tenant_name)
    tenant = (
        await session.execute(select(Tenant).where(Tenant.name == tenant_name))
    ).scalar_one_or_none()
    if tenant is None:
        tenant = Tenant(name=tenant_name)
        session.add(tenant)
        await session.commit()
    key, secret = await create_api_key(
        session, tenant.id, key_name, scopes=[ADMIN_TENANTS, ADMIN_KEYS]
    )
    return tenant, key, secret
