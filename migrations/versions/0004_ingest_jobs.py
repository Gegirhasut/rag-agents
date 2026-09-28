"""ingest_jobs, embed_batches, documents.job_id: фан-аут эмбеддинга по батчам (§8, §10.4)

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

job_status = pg.ENUM("pending", "running", "done", "failed", name="job_status", create_type=False)


def upgrade() -> None:
    job_status.create(op.get_bind(), checkfirst=True)
    op.create_table(
        "ingest_jobs",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column(
            "document_id",
            sa.Uuid,
            sa.ForeignKey(
                "documents.id", name="fk_ingest_jobs_document_id_documents", ondelete="CASCADE"
            ),
            nullable=False,
        ),
        sa.Column(
            "index_id",
            sa.Uuid,
            sa.ForeignKey(
                "agent_indexes.id", name="fk_ingest_jobs_index_id_agent_indexes", ondelete="CASCADE"
            ),
            nullable=False,
        ),
        sa.Column("kind", sa.Text, nullable=False, server_default="ingest"),
        sa.Column("status", job_status, nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("stats", pg.JSONB, nullable=False, server_default="{}"),
        sa.Column("error", sa.Text),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_ingest_jobs_document_id", "ingest_jobs", ["document_id"])
    op.create_table(
        "embed_batches",
        sa.Column(
            "job_id",
            sa.Uuid,
            sa.ForeignKey(
                "ingest_jobs.id", name="fk_embed_batches_job_id_ingest_jobs", ondelete="CASCADE"
            ),
            nullable=False,
        ),
        sa.Column("batch_no", sa.Integer, nullable=False),
        sa.Column("ord_from", sa.Integer, nullable=False),
        sa.Column("ord_to", sa.Integer, nullable=False),
        sa.Column("status", job_status, nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("embed_ms", sa.Integer),
        sa.Column("upsert_ms", sa.Integer),
        sa.PrimaryKeyConstraint("job_id", "batch_no", name="pk_embed_batches"),
    )
    # Текущая попытка обработки: задачи чужих (старых) попыток становятся no-op
    op.add_column("documents", sa.Column("job_id", sa.Uuid))


def downgrade() -> None:
    op.drop_column("documents", "job_id")
    op.drop_table("embed_batches")
    op.drop_index("ix_ingest_jobs_document_id", table_name="ingest_jobs")
    op.drop_table("ingest_jobs")
    job_status.drop(op.get_bind(), checkfirst=True)
