from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.errors import (
    ApiKeyExpiredError,
    ApiKeyRevokedError,
    InsufficientScopeError,
    InvalidApiKeyError,
    MissingApiKeyError,
    TenantDeactivatedError,
)
from app.auth.scopes import KNOWN_SCOPES, scope_allows
from app.auth.service import find_key_by_secret
from app.db.models import ApiKey
from app.db.session import get_session

# last_used_at is a convenience for humans, so it is refreshed at most this often per key
# rather than costing a database write on every request.
LAST_USED_REFRESH = timedelta(minutes=1)


@dataclass(frozen=True)
class AuthContext:
    """Who is calling. Holds no secret, so it is safe to log and pass around."""

    key_id: str
    key_prefix: str
    tenant_id: str
    tenant_name: str
    scopes: frozenset[str]
    allowed_models: tuple[str, ...] | None  # None: every model
    rpm_limit: int | None  # None: the gateway default
    tpm_limit: int | None  # None: the gateway default
    key_budget_nano: int | None  # monthly spending cap of the key, nano-dollars; None: no cap
    tenant_budget_nano: int | None  # monthly cap of the whole organisation


def _bearer_token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        raise MissingApiKeyError
    return token


async def _touch(session: AsyncSession, key: ApiKey) -> None:
    now = datetime.now(UTC)
    if key.last_used_at is not None and now - key.last_used_at < LAST_USED_REFRESH:
        return
    await session.execute(update(ApiKey).where(ApiKey.id == key.id).values(last_used_at=now))
    await session.commit()


async def authenticate(
    request: Request, session: Annotated[AsyncSession, Depends(get_session)]
) -> AuthContext:
    key = await find_key_by_secret(session, _bearer_token(request))
    if key is None:
        raise InvalidApiKeyError
    # Checked on every request, so a revocation takes effect immediately.
    status = key.status()
    if status == "revoked":
        raise ApiKeyRevokedError
    if status == "expired":
        raise ApiKeyExpiredError
    if not key.tenant.is_active:
        raise TenantDeactivatedError

    await _touch(session, key)
    context = AuthContext(
        key_id=key.id,
        key_prefix=key.key_prefix,
        tenant_id=key.tenant_id,
        tenant_name=key.tenant.name,
        scopes=frozenset(key.scopes),
        allowed_models=None if key.allowed_models is None else tuple(key.allowed_models),
        rpm_limit=key.rpm_limit,
        tpm_limit=key.tpm_limit,
        key_budget_nano=key.monthly_budget_nano_usd,
        tenant_budget_nano=key.tenant.monthly_budget_nano_usd,
    )
    request.state.auth = context  # lets the access log say who made the call
    return context


def require_scope(scope: str) -> Callable[..., Awaitable[AuthContext]]:
    """A dependency that authenticates the caller, then demands that their key holds ``scope``.

    Authentication failures stay 401; a valid key without the scope is a 403.
    """
    if scope not in KNOWN_SCOPES:
        raise ValueError(f"'{scope}' is not a known scope")  # fail at import, not at request time

    async def dependency(auth: Annotated[AuthContext, Depends(authenticate)]) -> AuthContext:
        if not scope_allows(auth.scopes, scope):
            raise InsufficientScopeError(scope)
        return auth

    return dependency
