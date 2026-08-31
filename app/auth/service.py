from collections.abc import Sequence
from datetime import datetime, timedelta
from enum import Enum

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.auth.keys import generate_key, hash_key, looks_like_key
from app.auth.model_access import normalize_allowed_models
from app.auth.scopes import DEFAULT_SCOPES, normalize_scopes
from app.db.models import ApiKey, Tenant, utcnow


class TenantNotFoundError(Exception):
    def __init__(self, tenant_id: str) -> None:
        super().__init__(f"tenant '{tenant_id}' does not exist")
        self.tenant_id = tenant_id


class TenantInactiveError(Exception):
    def __init__(self, tenant_id: str) -> None:
        super().__init__(f"tenant '{tenant_id}' is deactivated")
        self.tenant_id = tenant_id


class _Inherit(Enum):
    INHERIT = 0


INHERIT = _Inherit.INHERIT  # "keep the old key's setting", as opposed to None ("no restriction")


class KeyNotFoundError(Exception):
    """No such key for this tenant. Also raised for another tenant's key, revealing nothing."""

    def __init__(self, key_id: str) -> None:
        super().__init__(f"API key '{key_id}' does not exist")
        self.key_id = key_id


class KeyInactiveError(Exception):
    def __init__(self, key_id: str, status: str) -> None:
        super().__init__(f"API key '{key_id}' is {status} and cannot be rotated")
        self.key_id = key_id
        self.status = status


def _check_expiry(expires_at: datetime | None) -> None:
    if expires_at is None:
        return
    if expires_at.tzinfo is None:
        raise ValueError("expires_at must be timezone-aware")
    if expires_at <= utcnow():
        raise ValueError("expires_at must be in the future")


async def create_api_key(
    session: AsyncSession,
    tenant_id: str,
    name: str,
    expires_at: datetime | None = None,
    scopes: Sequence[str] | None = None,
    allowed_models: Sequence[str] | None = None,
) -> tuple[ApiKey, str]:
    """Create a key for a tenant. Returns the record and the plaintext secret.

    The secret exists only in this return value; the database keeps its hash. Without
    ``scopes`` the key gets the least-privilege defaults (chat and model listing); without
    ``allowed_models`` it may use every model.
    """
    _check_expiry(expires_at)
    granted = normalize_scopes(DEFAULT_SCOPES if scopes is None else scopes)
    models = None if allowed_models is None else normalize_allowed_models(allowed_models)
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
        tenant_id=tenant.id,
        name=name,
        key_prefix=generated.prefix,
        key_hash=generated.hash,
        expires_at=expires_at,
        scopes=granted,
        allowed_models=models,
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


async def _tenant_key(session: AsyncSession, tenant_id: str, key_id: str) -> ApiKey:
    key = await session.get(ApiKey, key_id)
    if key is None or key.tenant_id != tenant_id:
        raise KeyNotFoundError(key_id)
    return key


async def revoke_api_key(session: AsyncSession, tenant_id: str, key_id: str) -> ApiKey:
    """Disable a key for good. Takes effect on the next request; revoking twice is harmless."""
    key = await _tenant_key(session, tenant_id, key_id)
    if key.revoked_at is None:
        key.revoked_at = utcnow()
        await session.commit()
    return key


async def rotate_api_key(
    session: AsyncSession,
    tenant_id: str,
    key_id: str,
    *,
    grace: timedelta = timedelta(0),
    expires_at: datetime | None = None,
    scopes: Sequence[str] | None = None,
    allowed_models: Sequence[str] | None | _Inherit = INHERIT,
) -> tuple[ApiKey, str]:
    """Replace a key with a fresh one under the same name.

    The old key is revoked straight away, or, with ``grace``, keeps working for that long so
    clients can switch over (a grace period never lengthens an earlier expiry). The new key
    keeps the old key's scopes and model allow-list unless told otherwise (pass
    ``allowed_models=None`` to lift the restriction), but does not inherit its expiry; pass
    ``expires_at`` to give it one.
    """
    if grace < timedelta(0):
        raise ValueError("grace must not be negative")
    old = await _tenant_key(session, tenant_id, key_id)
    status = old.status()
    if status != "active":
        raise KeyInactiveError(key_id, status)

    models = old.allowed_models if isinstance(allowed_models, _Inherit) else allowed_models
    new_key, secret = await create_api_key(
        session,
        tenant_id,
        old.name,
        expires_at,
        list(old.scopes) if scopes is None else scopes,
        None if models is None else list(models),
    )
    if grace == timedelta(0):
        old.revoked_at = utcnow()
    else:
        cutoff = utcnow() + grace
        old.expires_at = cutoff if old.expires_at is None else min(old.expires_at, cutoff)
    await session.commit()
    return new_key, secret
