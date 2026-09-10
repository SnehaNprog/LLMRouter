from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import AuthContext, require_scope
from app.auth.errors import ScopeEscalationError
from app.auth.scopes import ADMIN_KEYS, ADMIN_TENANTS, can_grant
from app.auth.service import (
    INHERIT,
    create_api_key,
    get_api_key,
    list_api_keys,
    revoke_api_key,
    rotate_api_key,
)
from app.auth.tenants import (
    AdminConflictError,
    create_tenant,
    delete_tenant,
    get_tenant,
    list_tenants,
    update_tenant,
)
from app.db.session import get_session
from app.schemas.admin import (
    KeyCreate,
    KeyList,
    KeyOut,
    KeyRotate,
    KeyWithSecret,
    TenantCreate,
    TenantList,
    TenantOut,
    TenantUpdate,
)

tenant_admin = require_scope(ADMIN_TENANTS)
key_admin = require_scope(ADMIN_KEYS)

Session = Annotated[AsyncSession, Depends(get_session)]
Limit = Annotated[int, Query(ge=1, le=200)]
Offset = Annotated[int, Query(ge=0)]

tenants = APIRouter(prefix="/admin/tenants", tags=["admin"])
keys = APIRouter(prefix="/admin/tenants/{tenant_id}/keys", tags=["admin"])

_NO_STORE = {"Cache-Control": "no-store"}  # responses carrying a secret must not be cached


def _bad_request(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def _check_grantable(auth: AuthContext, scopes: list[str] | None) -> None:
    for scope in scopes or []:
        if not can_grant(auth.scopes, scope):
            raise ScopeEscalationError(scope)


# ---------------- tenants ----------------


@tenants.post("", status_code=201)
async def create_tenant_endpoint(
    body: TenantCreate, session: Session, _: Annotated[AuthContext, Depends(tenant_admin)]
) -> TenantOut:
    try:
        return TenantOut.of(await create_tenant(session, body.name))
    except ValueError as exc:
        raise _bad_request(exc) from None


@tenants.get("")
async def list_tenants_endpoint(
    session: Session,
    _: Annotated[AuthContext, Depends(tenant_admin)],
    limit: Limit = 50,
    offset: Offset = 0,
) -> TenantList:
    found = await list_tenants(session, limit=limit, offset=offset)
    return TenantList(data=[TenantOut.of(t) for t in found])


@tenants.get("/{tenant_id}")
async def get_tenant_endpoint(
    tenant_id: str, session: Session, _: Annotated[AuthContext, Depends(tenant_admin)]
) -> TenantOut:
    return TenantOut.of(await get_tenant(session, tenant_id))


@tenants.patch("/{tenant_id}")
async def update_tenant_endpoint(
    tenant_id: str,
    body: TenantUpdate,
    session: Session,
    auth: Annotated[AuthContext, Depends(tenant_admin)],
) -> TenantOut:
    if tenant_id == auth.tenant_id and body.is_active is False:
        raise AdminConflictError(
            "You cannot deactivate the tenant your own key belongs to.", "cannot_modify_own_tenant"
        )
    try:
        return TenantOut.of(
            await update_tenant(session, tenant_id, name=body.name, is_active=body.is_active)
        )
    except ValueError as exc:
        raise _bad_request(exc) from None


@tenants.delete("/{tenant_id}", status_code=204)
async def delete_tenant_endpoint(
    tenant_id: str, session: Session, auth: Annotated[AuthContext, Depends(tenant_admin)]
) -> Response:
    if tenant_id == auth.tenant_id:
        raise AdminConflictError(
            "You cannot delete the tenant your own key belongs to.", "cannot_modify_own_tenant"
        )
    await delete_tenant(session, tenant_id)
    return Response(status_code=204)


# ---------------- keys ----------------


@keys.post("", status_code=201)
async def create_key_endpoint(
    tenant_id: str,
    body: KeyCreate,
    session: Session,
    auth: Annotated[AuthContext, Depends(key_admin)],
    response: Response,
) -> KeyWithSecret:
    _check_grantable(auth, body.scopes)
    try:
        key, secret = await create_api_key(
            session,
            tenant_id,
            body.name,
            expires_at=body.expires_at,
            scopes=body.scopes,
            allowed_models=body.allowed_models,
            rpm_limit=body.rpm_limit,
        )
    except ValueError as exc:
        raise _bad_request(exc) from None
    response.headers.update(_NO_STORE)
    return KeyWithSecret(**KeyOut.of(key).model_dump(), secret=secret)


@keys.get("")
async def list_keys_endpoint(
    tenant_id: str,
    session: Session,
    _: Annotated[AuthContext, Depends(key_admin)],
    limit: Limit = 50,
    offset: Offset = 0,
) -> KeyList:
    found = await list_api_keys(session, tenant_id, limit=limit, offset=offset)
    return KeyList(data=[KeyOut.of(k) for k in found])


@keys.get("/{key_id}")
async def get_key_endpoint(
    tenant_id: str, key_id: str, session: Session, _: Annotated[AuthContext, Depends(key_admin)]
) -> KeyOut:
    return KeyOut.of(await get_api_key(session, tenant_id, key_id))


@keys.post("/{key_id}/revoke")
async def revoke_key_endpoint(
    tenant_id: str, key_id: str, session: Session, _: Annotated[AuthContext, Depends(key_admin)]
) -> KeyOut:
    return KeyOut.of(await revoke_api_key(session, tenant_id, key_id))


@keys.post("/{key_id}/rotate", status_code=201)
async def rotate_key_endpoint(
    tenant_id: str,
    key_id: str,
    body: KeyRotate,
    session: Session,
    auth: Annotated[AuthContext, Depends(key_admin)],
    response: Response,
) -> KeyWithSecret:
    # A rotated key inherits the old key's scopes unless told otherwise, so what it will end up
    # with is what must be grantable; otherwise rotating a broader admin's key would copy it.
    old = await get_api_key(session, tenant_id, key_id)
    _check_grantable(auth, body.scopes if body.scopes is not None else list(old.scopes))
    keep_models = "allowed_models" not in body.model_fields_set
    keep_rpm = "rpm_limit" not in body.model_fields_set
    try:
        key, secret = await rotate_api_key(
            session,
            tenant_id,
            key_id,
            grace=timedelta(seconds=body.grace_seconds),
            expires_at=body.expires_at,
            scopes=body.scopes,
            allowed_models=INHERIT if keep_models else body.allowed_models,
            rpm_limit=INHERIT if keep_rpm else body.rpm_limit,
        )
    except ValueError as exc:
        raise _bad_request(exc) from None
    response.headers.update(_NO_STORE)
    return KeyWithSecret(**KeyOut.of(key).model_dump(), secret=secret)
