"""ORM-модели (ARCHITECTURE §8). Наружу из repositories не выходят."""

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any
from uuid import UUID

from sqlalchemy import (
    CHAR,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, CITEXT, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from rag_agents.domain.enums import (
    DocumentStatus,
    IndexStatus,
    JobStatus,
    MessageRole,
    MessageStatus,
)
from rag_agents.models.base import Base, TimestampMixin, uuid7


def pg_enum[E: StrEnum](enum_cls: type[E], name: str) -> Enum:
    return Enum(enum_cls, name=name, values_callable=lambda e: [m.value for m in e])


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid7)
    email: Mapped[str] = mapped_column(CITEXT, unique=True)
    # NULL = войти по паролю нельзя (seed-пользователь, пока не задан пароль через CLI)
    password_hash: Mapped[str | None] = mapped_column(Text)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")


class ApiKey(TimestampMixin, Base):
    __tablename__ = "api_keys"
    __table_args__ = (Index("ix_api_keys_user_id", "user_id"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid7)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(Text)
    # Открытая часть ключа rag_<prefix>_<secret>: по ней ищем строку, секрет — только sha256
    prefix: Mapped[str] = mapped_column(CHAR(8), unique=True)
    secret_hash: Mapped[bytes] = mapped_column(LargeBinary)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Agent(TimestampMixin, Base):
    __tablename__ = "agents"
    __table_args__ = (
        UniqueConstraint("owner_id", "slug"),
        Index("ix_agents_owner_alive", "owner_id", postgresql_where="deleted_at IS NULL"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid7)
    owner_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(Text)
    slug: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text, default="", server_default="")
    persona_prompt: Mapped[str | None] = mapped_column(Text)
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    # Циклическая ссылка agents ↔ agent_indexes: FK отложенный,
    # обе строки вставляются в одной транзакции
    active_index_id: Mapped[UUID | None] = mapped_column(
        ForeignKey(
            "agent_indexes.id",
            name="fk_agents_active_index_id_agent_indexes",
            deferrable=True,
            initially="DEFERRED",
            use_alter=True,
        )
    )
    corpus_version: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AgentIndex(TimestampMixin, Base):
    __tablename__ = "agent_indexes"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid7)
    agent_id: Mapped[UUID] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"))
    embedding_model: Mapped[str] = mapped_column(Text)
    dim: Mapped[int] = mapped_column(Integer)
    collection: Mapped[str] = mapped_column(Text)
    chunking_version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    status: Mapped[IndexStatus] = mapped_column(pg_enum(IndexStatus, "index_status"))
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Document(TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("agent_id", "sha256"),
        Index("ix_documents_agent_status", "agent_id", "status"),
        Index(
            "ix_documents_inflight",
            "status",
            "heartbeat_at",
            postgresql_where="status IN ('queued', 'processing')",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid7)
    agent_id: Mapped[UUID] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(Text)
    format: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[bytes] = mapped_column(LargeBinary)
    storage_key: Mapped[str] = mapped_column(Text)
    status: Mapped[DocumentStatus] = mapped_column(
        pg_enum(DocumentStatus, "document_status"),
        default=DocumentStatus.QUEUED,
        server_default=DocumentStatus.QUEUED.value,
    )
    stage: Mapped[str | None] = mapped_column(Text)
    error_code: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    chunks_total: Mapped[int | None] = mapped_column(Integer)
    batches_total: Mapped[int | None] = mapped_column(Integer)
    batches_done: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Текущая попытка (ingest_jobs.id): задачи прежних попыток — no-op. Без FK: цикл с jobs
    job_id: Mapped[UUID | None] = mapped_column()
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class IngestJob(TimestampMixin, Base):
    """Попытка обработать документ в конкретный индекс (ARCHITECTURE §8)."""

    __tablename__ = "ingest_jobs"
    __table_args__ = (Index("ix_ingest_jobs_document_id", "document_id"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid7)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    index_id: Mapped[UUID] = mapped_column(ForeignKey("agent_indexes.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(Text, default="ingest", server_default="ingest")
    status: Mapped[JobStatus] = mapped_column(
        pg_enum(JobStatus, "job_status"), default=JobStatus.PENDING, server_default="pending"
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    stats: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    error: Mapped[str | None] = mapped_column(Text)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EmbedBatch(Base):
    """Идемпотентный учёт батчей эмбеддинга: done ставит только первый исполнитель."""

    __tablename__ = "embed_batches"

    job_id: Mapped[UUID] = mapped_column(
        ForeignKey("ingest_jobs.id", ondelete="CASCADE"), primary_key=True
    )
    batch_no: Mapped[int] = mapped_column(Integer, primary_key=True)
    ord_from: Mapped[int] = mapped_column(Integer)
    ord_to: Mapped[int] = mapped_column(Integer)
    status: Mapped[JobStatus] = mapped_column(
        pg_enum(JobStatus, "job_status"), default=JobStatus.PENDING, server_default="pending"
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    embed_ms: Mapped[int | None] = mapped_column(Integer)
    upsert_ms: Mapped[int | None] = mapped_column(Integer)


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "ord"),
        Index("ix_chunks_agent_id", "agent_id"),
    )

    # = id точки в Qdrant (uuid5, детерминирован → идемпотентный upsert)
    id: Mapped[UUID] = mapped_column(primary_key=True)
    agent_id: Mapped[UUID] = mapped_column()
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    ord: Mapped[int] = mapped_column(Integer)
    section_path: Mapped[list[str]] = mapped_column(ARRAY(Text))
    chapter_title: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    embed_text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    page_from: Mapped[int | None] = mapped_column(Integer)
    page_to: Mapped[int | None] = mapped_column(Integer)
    content_hash: Mapped[bytes] = mapped_column(LargeBinary)


class Chat(TimestampMixin, Base):
    __tablename__ = "chats"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid7)
    agent_id: Mapped[UUID] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"))
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    title: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Message(TimestampMixin, Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_chat_created", "chat_id", "created_at"),
        CheckConstraint("role IN ('user', 'assistant')", name="role"),
        CheckConstraint("feedback IS NULL OR feedback IN (-1, 1)", name="feedback"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid7)
    chat_id: Mapped[UUID] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"))
    role: Mapped[MessageRole] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text, default="", server_default="")
    status: Mapped[MessageStatus] = mapped_column(pg_enum(MessageStatus, "message_status"))
    standalone_question: Mapped[str | None] = mapped_column(Text)
    citations: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    refused: Mapped[bool | None] = mapped_column(Boolean)
    grounded: Mapped[bool | None] = mapped_column(Boolean)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    prompt_version: Mapped[str | None] = mapped_column(Text)
    index_id: Mapped[UUID | None] = mapped_column()
    trace_id: Mapped[str | None] = mapped_column(Text)
    # Оценка пользователя: 1 = 👍, -1 = 👎. Дублируется score-ом в Langfuse
    feedback: Mapped[int | None] = mapped_column(SmallInteger)


class EvalRun(Base):
    """Прогон eval: датасет × конфигурация; metrics — сводка после завершения (§15)."""

    __tablename__ = "eval_runs"
    __table_args__ = (Index("ix_eval_runs_agent_id", "agent_id", "started_at"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid7)
    agent_id: Mapped[UUID] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"))
    dataset: Mapped[str] = mapped_column(Text)
    dataset_sha: Mapped[str] = mapped_column(Text)
    config_name: Mapped[str] = mapped_column(Text)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EvalItem(Base):
    __tablename__ = "eval_items"

    run_id: Mapped[UUID] = mapped_column(
        ForeignKey("eval_runs.id", ondelete="CASCADE"), primary_key=True
    )
    item_id: Mapped[str] = mapped_column(Text, primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(Text)
    retrieved: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    answer: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    refused: Mapped[bool | None] = mapped_column(Boolean)
    scores: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    judge: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    trace_id: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    ttft_ms: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
