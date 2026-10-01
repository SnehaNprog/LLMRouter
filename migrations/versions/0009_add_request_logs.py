"""add request logs

Revision ID: 0009
Revises: 0008
Create Date: 2026-11-04 09:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "request_logs",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("day", sa.String(length=10), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("tenant_id", sa.String(length=32), nullable=False),
        sa.Column("key_id", sa.String(length=32), nullable=False),
        sa.Column("requested_model", sa.String(length=200), nullable=False),
        sa.Column("served_model", sa.String(length=200), nullable=True),
        sa.Column("deployment_id", sa.String(length=32), nullable=True),
        sa.Column("status_code", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("stream", sa.Boolean(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_nano_usd", sa.BigInteger(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("routed", sa.Boolean(), nullable=False),
        sa.Column("tier", sa.String(length=16), nullable=True),
        sa.Column("escalated", sa.Boolean(), nullable=False),
        sa.Column("complexity", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_request_logs_created_at", "request_logs", ["created_at"])
    op.create_index("ix_request_logs_tenant_created", "request_logs", ["tenant_id", "created_at"])
    op.create_index("ix_request_logs_key_created", "request_logs", ["key_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_request_logs_key_created", table_name="request_logs")
    op.drop_index("ix_request_logs_tenant_created", table_name="request_logs")
    op.drop_index("ix_request_logs_created_at", table_name="request_logs")
    op.drop_table("request_logs")
