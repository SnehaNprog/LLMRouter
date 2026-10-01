import json
import uuid
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import JSON, BigInteger, Float, ForeignKey, Index, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.auth.scopes import DEFAULT_SCOPES
from app.db.base import Base, UTCDateTime


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(UTC)


class Tenant(Base):
    """An organisation using the gateway. Everything a tenant owns is isolated from the rest."""

    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    # Calendar-month spending cap for the whole organisation, in nano-dollars; NULL: no cap.
    monthly_budget_nano_usd: Mapped[int | None] = mapped_column(BigInteger, default=None)
    # Smart-routing policy for this organisation's "auto" requests; NULL means no restriction.
    routing_pinned_model: Mapped[str | None] = mapped_column(String(200), default=None)
    routing_max_tier: Mapped[str | None] = mapped_column(String(16), default=None)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    api_keys: Mapped[list["ApiKey"]] = relationship(
        back_populates="tenant", cascade="all, delete-orphan", passive_deletes=True
    )


class ApiKey(Base):
    """A virtual API key. Only a hash of the secret is stored, never the secret itself."""

    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    key_prefix: Mapped[str] = mapped_column(String(16))  # shown in UIs so keys can be recognised
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)
    # What the key may do (see app/auth/scopes.py). Existing rows get the defaults.
    scopes: Mapped[list[str]] = mapped_column(
        JSON,
        default=lambda: list(DEFAULT_SCOPES),
        server_default=text(f"'{json.dumps(list(DEFAULT_SCOPES))}'"),
    )
    # Model aliases (or "fast*" style patterns) the key may use; NULL means every model.
    allowed_models: Mapped[list[str] | None] = mapped_column(JSON(none_as_null=True), default=None)
    # Requests per minute this key may make; NULL falls back to the gateway-wide default.
    rpm_limit: Mapped[int | None] = mapped_column(default=None)
    # Tokens per minute (prompt plus completion); NULL falls back to the gateway-wide default.
    tpm_limit: Mapped[int | None] = mapped_column(default=None)
    # Calendar-month spending cap for this key, in nano-dollars; NULL: no cap.
    monthly_budget_nano_usd: Mapped[int | None] = mapped_column(BigInteger, default=None)
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)
    revoked_at: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)

    tenant: Mapped[Tenant] = relationship(back_populates="api_keys")

    def status(self, now: datetime | None = None) -> Literal["active", "expired", "revoked"]:
        """Revocation wins over expiry; a key is expired from its expiry instant onwards."""
        now = now or utcnow()
        if self.revoked_at is not None:
            return "revoked"
        if self.expires_at is not None and now >= self.expires_at:
            return "expired"
        return "active"


class RequestLog(Base):
    """One request that reached the model-serving stage: who, which model, how it went, its cost.

    Only metadata is kept. Prompts and replies are never stored, so the log cannot leak what
    people asked or were told, and costs nothing to keep safe. Requests turned away earlier (an
    unknown key, a missing scope, the per-minute request limit) are not here; the access log
    has them.
    """

    __tablename__ = "request_logs"
    __table_args__ = (
        Index("ix_request_logs_tenant_created", "tenant_id", "created_at"),
        Index("ix_request_logs_key_created", "key_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    day: Mapped[str] = mapped_column(String(10))  # "YYYY-MM-DD" (UTC), to group by day portably
    request_id: Mapped[str | None] = mapped_column(String(64), default=None)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    key_id: Mapped[str] = mapped_column(String(32))
    requested_model: Mapped[str] = mapped_column(String(200))  # as the caller wrote it ("auto"...)
    served_model: Mapped[str | None] = mapped_column(String(200), default=None)  # the alias used
    deployment_id: Mapped[str | None] = mapped_column(String(32), default=None)
    status_code: Mapped[int]
    error_code: Mapped[str | None] = mapped_column(String(64), default=None)
    stream: Mapped[bool] = mapped_column(default=False)
    prompt_tokens: Mapped[int] = mapped_column(default=0)
    completion_tokens: Mapped[int] = mapped_column(default=0)
    cost_nano_usd: Mapped[int] = mapped_column(BigInteger, default=0)
    latency_ms: Mapped[int] = mapped_column(default=0)
    # Smart routing: set when the request asked for "auto".
    routed: Mapped[bool] = mapped_column(default=False)
    tier: Mapped[str | None] = mapped_column(String(16), default=None)  # tier that served it
    escalated: Mapped[bool] = mapped_column(default=False)
    complexity: Mapped[float | None] = mapped_column(Float, default=None)
