from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.tiers import Tier
from app.db.models import ApiKey, Tenant
from app.quota.budget import nano_to_usd


def _usd(nano: int | None) -> float | None:
    return None if nano is None else nano_to_usd(nano)


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a misspelt field is an error, not silently ignored


class TenantCreate(_Request):
    name: str = Field(min_length=1, max_length=200)
    monthly_budget_usd: float | None = Field(
        default=None, ge=0.000001, le=1_000_000
    )  # default: no cap
    pinned_model: str | None = Field(default=None, min_length=1, max_length=200)
    max_tier: Tier | None = None


class TenantUpdate(_Request):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    is_active: bool | None = None
    monthly_budget_usd: float | None = Field(
        default=None, ge=0.000001, le=1_000_000
    )  # omit to keep; null removes the cap
    pinned_model: str | None = Field(default=None, min_length=1, max_length=200)  # omit to keep
    max_tier: Tier | None = None  # omit to keep; null removes the cap


class TenantOut(BaseModel):
    id: str
    name: str
    is_active: bool
    monthly_budget_usd: float | None
    pinned_model: str | None
    max_tier: Tier | None
    created_at: datetime

    @classmethod
    def of(cls, tenant: Tenant) -> "TenantOut":
        return cls(
            id=tenant.id,
            name=tenant.name,
            is_active=tenant.is_active,
            monthly_budget_usd=_usd(tenant.monthly_budget_nano_usd),
            pinned_model=tenant.routing_pinned_model,
            max_tier=None if tenant.routing_max_tier is None else Tier(tenant.routing_max_tier),
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
    tpm_limit: int | None = Field(default=None, ge=1, le=1_000_000_000)  # default: gateway default
    monthly_budget_usd: float | None = Field(
        default=None, ge=0.000001, le=1_000_000
    )  # default: no cap
    expires_at: datetime | None = None


class KeyRotate(_Request):
    grace_seconds: int = Field(default=0, ge=0, le=30 * 24 * 3600)
    expires_at: datetime | None = None
    scopes: list[str] | None = None  # default: keep the old key's scopes
    allowed_models: list[str] | None = None  # omit to keep; null lifts the restriction
    rpm_limit: int | None = Field(default=None, ge=1, le=1_000_000)  # omit to keep; null: default
    tpm_limit: int | None = Field(
        default=None, ge=1, le=1_000_000_000
    )  # omit to keep; null: default
    monthly_budget_usd: float | None = Field(
        default=None, ge=0.000001, le=1_000_000
    )  # omit to keep; null removes the cap


class KeyOut(BaseModel):
    id: str
    tenant_id: str
    name: str
    key_prefix: str
    scopes: list[str]
    allowed_models: list[str] | None
    rpm_limit: int | None
    tpm_limit: int | None
    monthly_budget_usd: float | None
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
            tpm_limit=key.tpm_limit,
            monthly_budget_usd=_usd(key.monthly_budget_nano_usd),
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
