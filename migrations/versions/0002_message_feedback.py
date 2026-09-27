"""messages.feedback: оценка ответа пользователем (👍 = 1, 👎 = -1)

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("feedback", sa.SmallInteger, nullable=True))
    op.create_check_constraint(
        "ck_messages_feedback", "messages", "feedback IS NULL OR feedback IN (-1, 1)"
    )


def downgrade() -> None:
    op.drop_constraint("ck_messages_feedback", "messages", type_="check")
    op.drop_column("messages", "feedback")
