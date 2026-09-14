"""add monthly budgets

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-02 03:09:54.200028
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("api_keys", schema=None) as batch_op:
        batch_op.add_column(sa.Column("monthly_budget_nano_usd", sa.BigInteger(), nullable=True))

    with op.batch_alter_table("tenants", schema=None) as batch_op:
        batch_op.add_column(sa.Column("monthly_budget_nano_usd", sa.BigInteger(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("tenants", schema=None) as batch_op:
        batch_op.drop_column("monthly_budget_nano_usd")

    with op.batch_alter_table("api_keys", schema=None) as batch_op:
        batch_op.drop_column("monthly_budget_nano_usd")
