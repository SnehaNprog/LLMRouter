"""add tenant routing policy

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-02 04:27:22.251185
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("tenants", schema=None) as batch_op:
        batch_op.add_column(sa.Column("routing_pinned_model", sa.String(length=200), nullable=True))
        batch_op.add_column(sa.Column("routing_max_tier", sa.String(length=16), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("tenants", schema=None) as batch_op:
        batch_op.drop_column("routing_max_tier")
        batch_op.drop_column("routing_pinned_model")
