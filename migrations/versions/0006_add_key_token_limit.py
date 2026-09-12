"""add key token limit

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-02 02:59:26.073813
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("api_keys", schema=None) as batch_op:
        batch_op.add_column(sa.Column("tpm_limit", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("api_keys", schema=None) as batch_op:
        batch_op.drop_column("tpm_limit")
