"""initial schema: users, agents, agent_indexes, documents, chunks, chats, messages

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

index_status = pg.ENUM("building", "active", "retired", name="index_status", create_type=False)
document_status = pg.ENUM(
    "queued", "processing", "done", "failed", "deleting", name="document_status", create_type=False
)
message_status = pg.ENUM(
    "pending", "streaming", "done", "error", "cancelled", name="message_status", create_type=False
)


def ts(name: str, *, nullable: bool = True, now: bool = False) -> sa.Column:  # type: ignore[type-arg]
    return sa.Column(
        name,
        sa.DateTime(timezone=True),
        nullable=nullable,
        server_default=sa.func.now() if now else None,
    )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS citext")
    for enum in (index_status, document_status, message_status):
        enum.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "users",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column("email", pg.CITEXT, nullable=False),
        sa.Column("password_hash", sa.Text),
        sa.Column("is_admin", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        ts("created_at", nullable=False, now=True),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )

    op.create_table(
        "agents",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column(
            "owner_id",
            sa.Uuid,
            sa.ForeignKey("users.id", name="fk_agents_owner_id_users"),
            nullable=False,
        ),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("slug", sa.Text, nullable=False),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        sa.Column("persona_prompt", sa.Text),
        sa.Column("settings", pg.JSONB, nullable=False, server_default="{}"),
        sa.Column("active_index_id", sa.Uuid),
        sa.Column("corpus_version", sa.Integer, nullable=False, server_default="0"),
        ts("created_at", nullable=False, now=True),
        ts("updated_at", nullable=False, now=True),
        ts("deleted_at"),
        sa.UniqueConstraint("owner_id", "slug", name="uq_agents_owner_id"),
    )
    op.create_index(
        "ix_agents_owner_alive",
        "agents",
        ["owner_id"],
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    op.create_table(
        "agent_indexes",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column(
            "agent_id",
            sa.Uuid,
            sa.ForeignKey("agents.id", ondelete="CASCADE", name="fk_agent_indexes_agent_id_agents"),
            nullable=False,
        ),
        sa.Column("embedding_model", sa.Text, nullable=False),
        sa.Column("dim", sa.Integer, nullable=False),
        sa.Column("collection", sa.Text, nullable=False),
        sa.Column("chunking_version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("status", index_status, nullable=False),
        ts("created_at", nullable=False, now=True),
        ts("activated_at"),
    )
    # Циклическая ссылка agents ↔ agent_indexes: FK отложенный, чтобы создать обе строки в одной транзакции
    op.create_foreign_key(
        "fk_agents_active_index_id_agent_indexes",
        "agents",
        "agent_indexes",
        ["active_index_id"],
        ["id"],
        deferrable=True,
        initially="DEFERRED",
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column(
            "agent_id",
            sa.Uuid,
            sa.ForeignKey("agents.id", ondelete="CASCADE", name="fk_documents_agent_id_agents"),
            nullable=False,
        ),
        sa.Column("filename", sa.Text, nullable=False),
        sa.Column("format", sa.Text, nullable=False),
        sa.Column("size_bytes", sa.BigInteger, nullable=False),
        sa.Column("sha256", sa.LargeBinary, nullable=False),
        sa.Column("storage_key", sa.Text, nullable=False),
        sa.Column("status", document_status, nullable=False, server_default="queued"),
        sa.Column("stage", sa.Text),
        sa.Column("error_code", sa.Text),
        sa.Column("error_message", sa.Text),
        sa.Column("title", sa.Text),
        sa.Column("author", sa.Text),
        sa.Column("meta", pg.JSONB, nullable=False, server_default="{}"),
        sa.Column("chunks_total", sa.Integer),
        sa.Column("batches_total", sa.Integer),
        sa.Column("batches_done", sa.Integer, nullable=False, server_default="0"),
        ts("heartbeat_at"),
        ts("created_at", nullable=False, now=True),
        ts("started_at"),
        ts("finished_at"),
        sa.UniqueConstraint("agent_id", "sha256", name="uq_documents_agent_id"),
    )
    op.create_index("ix_documents_agent_status", "documents", ["agent_id", "status"])
    op.create_index(
        "ix_documents_inflight",
        "documents",
        ["status", "heartbeat_at"],
        postgresql_where=sa.text("status IN ('queued', 'processing')"),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column("agent_id", sa.Uuid, nullable=False),
        sa.Column(
            "document_id",
            sa.Uuid,
            sa.ForeignKey(
                "documents.id", ondelete="CASCADE", name="fk_chunks_document_id_documents"
            ),
            nullable=False,
        ),
        sa.Column("ord", sa.Integer, nullable=False),
        sa.Column("section_path", pg.ARRAY(sa.Text), nullable=False),
        sa.Column("chapter_title", sa.Text),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("embed_text", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("char_start", sa.Integer, nullable=False),
        sa.Column("char_end", sa.Integer, nullable=False),
        sa.Column("page_from", sa.Integer),
        sa.Column("page_to", sa.Integer),
        sa.Column("content_hash", sa.LargeBinary, nullable=False),
        sa.UniqueConstraint("document_id", "ord", name="uq_chunks_document_id"),
    )
    op.create_index("ix_chunks_agent_id", "chunks", ["agent_id"])

    op.create_table(
        "chats",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column(
            "agent_id",
            sa.Uuid,
            sa.ForeignKey("agents.id", ondelete="CASCADE", name="fk_chats_agent_id_agents"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.Uuid,
            sa.ForeignKey("users.id", name="fk_chats_user_id_users"),
            nullable=False,
        ),
        sa.Column("title", sa.Text),
        sa.Column("summary", sa.Text),
        ts("created_at", nullable=False, now=True),
        ts("updated_at", nullable=False, now=True),
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column(
            "chat_id",
            sa.Uuid,
            sa.ForeignKey("chats.id", ondelete="CASCADE", name="fk_messages_chat_id_chats"),
            nullable=False,
        ),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text, nullable=False, server_default=""),
        sa.Column("status", message_status, nullable=False),
        sa.Column("standalone_question", sa.Text),
        sa.Column("citations", pg.JSONB),
        sa.Column("refused", sa.Boolean),
        sa.Column("grounded", sa.Boolean),
        sa.Column("usage", pg.JSONB),
        sa.Column("prompt_version", sa.Text),
        sa.Column("index_id", sa.Uuid),
        sa.Column("trace_id", sa.Text),
        ts("created_at", nullable=False, now=True),
        sa.CheckConstraint("role IN ('user', 'assistant')", name="ck_messages_role"),
    )
    op.create_index("ix_messages_chat_created", "messages", ["chat_id", "created_at"])


def downgrade() -> None:
    for table in ("messages", "chats", "chunks", "documents"):
        op.drop_table(table)
    op.drop_constraint("fk_agents_active_index_id_agent_indexes", "agents", type_="foreignkey")
    for table in ("agent_indexes", "agents", "users"):
        op.drop_table(table)
    for enum in (message_status, document_status, index_status):
        enum.drop(op.get_bind(), checkfirst=True)
