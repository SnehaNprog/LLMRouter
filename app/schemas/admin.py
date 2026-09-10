from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import ApiKey, Tenant


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a misspelt field is an error, not silently ignored


class TenantCreate(_Request):
    name: str = Field(min_length=1, max_length=200)


class TenantUpdate(_Request):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    is_active: bool | None = None


class TenantOut(BaseModel):
    id: str
    name: str
    is_active: bool
    created_at: datetime

    @classmethod
    def of(cls, tenant: Tenant) -> "TenantOut":
        return cls(
            id=tenant.id,
            name=tenant.name,
            is_active=tenant.is_active,
            created_at=tenant.created_at,
        )


class TenantList(BaseModel):
    object: Literal["list"] = "list"
    data: list[TenantOut]


class KeyCreate(_Request):
    name: str = Field(min_length=1, max_length=200)
    scopes: list[str] | None = None  # default: chat and model listing
    allowed_models: list[str] | None = None  # default: every model
    rpm_limit: int | None = Field(default=None, ge=1, le=1_000_000)  # default: gateway default
    expires_at: datetime | None = None


class KeyRotate(_Request):
    grace_seconds: int = Field(default=0, ge=0, le=30 * 24 * 3600)
    expires_at: datetime | None = None
    scopes: list[str] | None = None  # default: keep the old key's scopes
    allowed_models: list[str] | None = None  # omit to keep; null lifts the restriction
    rpm_limit: int | None = Field(default=None, ge=1, le=1_000_000)  # omit to keep; null: default


class KeyOut(BaseModel):
    id: str
    tenant_id: str
    name: str
    key_prefix: str
    scopes: list[str]
    allowed_models: list[str] | None
    rpm_limit: int | None
    status: Literal["active", "expired", "revoked"]
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None
    revoked_at: datetime | None

    @classmethod
    def of(cls, key: ApiKey) -> "KeyOut":
        return cls(
            id=key.id,
            tenant_id=key.tenant_id,
            name=key.name,
            key_prefix=key.key_prefix,
            scopes=list(key.scopes),
            allowed_models=None if key.allowed_models is None else list(key.allowed_models),
            rpm_limit=key.rpm_limit,
            status=key.status(),
            created_at=key.created_at,
            last_used_at=key.last_used_at,
            expires_at=key.expires_at,
            revoked_at=key.revoked_at,
        )


class KeyWithSecret(KeyOut):
    secret: str = Field(description="The API key. Shown once; it cannot be retrieved later.")


class KeyList(BaseModel):
    object: Literal["list"] = "list"
    data: list[KeyOut]
