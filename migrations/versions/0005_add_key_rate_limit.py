"""add key rate limit

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-02 02:49:32.336346
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("api_keys", schema=None) as batch_op:
        batch_op.add_column(sa.Column("rpm_limit", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("api_keys", schema=None) as batch_op:
        batch_op.drop_column("rpm_limit")
