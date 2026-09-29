"""eval_runs, eval_items: прогоны eval и результаты по вопросам (ARCHITECTURE §8, §15)

Revision ID: 0005
Revises: 0004
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "eval_runs",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column(
            "agent_id",
            sa.Uuid,
            sa.ForeignKey("agents.id", name="fk_eval_runs_agent_id_agents", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("dataset", sa.Text, nullable=False),
        sa.Column("dataset_sha", sa.Text, nullable=False),
        sa.Column("config_name", sa.Text, nullable=False),
        sa.Column("config", pg.JSONB, nullable=False),
        sa.Column("metrics", pg.JSONB),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_eval_runs_agent_id", "eval_runs", ["agent_id", "started_at"])
    op.create_table(
        "eval_items",
        sa.Column(
            "run_id",
            sa.Uuid,
            sa.ForeignKey(
                "eval_runs.id", name="fk_eval_items_run_id_eval_runs", ondelete="CASCADE"
            ),
            primary_key=True,
        ),
        sa.Column("item_id", sa.Text, primary_key=True),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("category", sa.Text, nullable=False),
        sa.Column("retrieved", pg.JSONB),
        sa.Column("answer", sa.Text),
        sa.Column("error", sa.Text),
        sa.Column("refused", sa.Boolean),
        sa.Column("scores", pg.JSONB),
        sa.Column("judge", pg.JSONB),
        sa.Column("trace_id", sa.Text),
        sa.Column("latency_ms", sa.Integer),
        sa.Column("ttft_ms", sa.Integer),
        sa.Column("cost_usd", sa.Numeric(10, 6)),
    )


def downgrade() -> None:
    op.drop_table("eval_items")
    op.drop_table("eval_runs")
