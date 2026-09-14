import json
import uuid
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import JSON, BigInteger, ForeignKey, String, text
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
