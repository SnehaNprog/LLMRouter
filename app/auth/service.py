from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.auth.keys import generate_key, hash_key, looks_like_key
from app.db.models import ApiKey, Tenant


class TenantNotFoundError(Exception):
    def __init__(self, tenant_id: str) -> None:
        super().__init__(f"tenant '{tenant_id}' does not exist")
        self.tenant_id = tenant_id


class TenantInactiveError(Exception):
    def __init__(self, tenant_id: str) -> None:
        super().__init__(f"tenant '{tenant_id}' is deactivated")
        self.tenant_id = tenant_id


async def create_api_key(session: AsyncSession, tenant_id: str, name: str) -> tuple[ApiKey, str]:
    """Create a key for a tenant. Returns the record and the plaintext secret.

    The secret exists only in this return value; the database keeps its hash.
    """
    name = name.strip()
    if not name:
        raise ValueError("a key needs a name")
    if len(name) > 200:
        raise ValueError("key names are limited to 200 characters")

    tenant = await session.get(Tenant, tenant_id)
    if tenant is None:
        raise TenantNotFoundError(tenant_id)
    if not tenant.is_active:
        raise TenantInactiveError(tenant_id)

    generated = generate_key()
    key = ApiKey(
        tenant_id=tenant.id, name=name, key_prefix=generated.prefix, key_hash=generated.hash
    )
    session.add(key)
    await session.commit()
    return key, generated.secret


async def find_key_by_secret(session: AsyncSession, secret: str) -> ApiKey | None:
    """Look a presented key up by its hash. The key's tenant is loaded with it."""
    if not looks_like_key(secret):
        return None
    result = await session.execute(
        select(ApiKey).options(joinedload(ApiKey.tenant)).where(ApiKey.key_hash == hash_key(secret))
    )
    return result.scalar_one_or_none()
