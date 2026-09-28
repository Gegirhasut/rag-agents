from datetime import timedelta
from uuid import UUID

from sqlalchemy import or_, select, type_coerce, update
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agents.core.ids import utcnow
from rag_agents.domain.documents import DocumentOut
from rag_agents.domain.enums import DocumentStatus, IngestStage
from rag_agents.models.entities import Document

STALE_AFTER = timedelta(minutes=10)


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def insert_if_new(
        self,
        *,
        document_id: UUID,
        agent_id: UUID,
        filename: str,
        fmt: str,
        size_bytes: int,
        sha256: bytes,
        storage_key: str,
    ) -> tuple[DocumentOut, bool]:
        """Дедупликация по (agent_id, sha256). Возвращает (документ, создан_ли)."""
        stmt = (
            insert(Document)
            .values(
                id=document_id,
                agent_id=agent_id,
                filename=filename,
                format=fmt,
                size_bytes=size_bytes,
                sha256=sha256,
                storage_key=storage_key,
                status=DocumentStatus.QUEUED,
            )
            .on_conflict_do_nothing(index_elements=["agent_id", "sha256"])
            .returning(Document.id)
        )
        created_id = await self.s.scalar(stmt)
        if created_id is not None:
            doc = await self.s.get_one(Document, created_id)
            return DocumentOut.model_validate(doc), True
        existing = await self.s.scalar(
            select(Document).where(Document.agent_id == agent_id, Document.sha256 == sha256)
        )
        if existing is None:  # pragma: no cover — конфликт без строки невозможен
            raise RuntimeError("dedup conflict without existing row")
        return DocumentOut.model_validate(existing), False

    async def list_for_agent(self, agent_id: UUID) -> list[DocumentOut]:
        rows = await self.s.scalars(
            select(Document)
            .where(Document.agent_id == agent_id, Document.status != DocumentStatus.DELETING)
            .order_by(Document.created_at.desc(), Document.id.desc())
        )
        return [DocumentOut.model_validate(d) for d in rows]

    async def get(self, agent_id: UUID, document_id: UUID) -> DocumentOut | None:
        doc = await self.s.scalar(
            select(Document).where(Document.id == document_id, Document.agent_id == agent_id)
        )
        return DocumentOut.model_validate(doc) if doc else None

    async def get_for_worker(self, document_id: UUID) -> DocumentOut | None:
        doc = await self.s.get(Document, document_id)
        return DocumentOut.model_validate(doc) if doc else None

    async def claim(self, document_id: UUID) -> bool:
        """Условный UPDATE вместо SELECT-then-UPDATE: повторная доставка задачи — no-op."""
        now = utcnow()
        result = await self.s.execute(
            update(Document)
            .where(
                Document.id == document_id,
                or_(
                    Document.status == DocumentStatus.QUEUED,
                    (Document.status == DocumentStatus.PROCESSING)
                    & (Document.heartbeat_at < now - STALE_AFTER),
                ),
            )
            .values(
                status=DocumentStatus.PROCESSING,
                stage=IngestStage.PARSING.value,
                heartbeat_at=now,
                started_at=now,
                error_code=None,
                error_message=None,
            )
            .returning(Document.id)
        )
        return result.scalar_one_or_none() is not None

    async def set_stage(
        self,
        document_id: UUID,
        stage: IngestStage,
        *,
        title: str | None = None,
        author: str | None = None,
        meta: dict[str, object] | None = None,
        chunks_total: int | None = None,
    ) -> None:
        values: dict[str, object] = {"stage": stage.value, "heartbeat_at": utcnow()}
        if title is not None:
            values["title"] = title
        if author is not None:
            values["author"] = author
        if meta is not None:
            values["meta"] = meta
        if chunks_total is not None:
            values["chunks_total"] = chunks_total
        await self.s.execute(update(Document).where(Document.id == document_id).values(**values))

    async def heartbeat(self, document_id: UUID) -> None:
        await self.s.execute(
            update(Document).where(Document.id == document_id).values(heartbeat_at=utcnow())
        )

    async def mark_done(self, document_id: UUID, timings: dict[str, int] | None = None) -> None:
        """timings (мс по шагам) дописываются в meta.timings, остальной meta не трогаем."""
        patch = type_coerce({"timings": timings or {}}, JSONB)
        await self.s.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(
                status=DocumentStatus.DONE,
                stage=None,
                finished_at=utcnow(),
                meta=Document.meta.op("||")(patch),
            )
        )

    async def mark_failed(self, document_id: UUID, code: str, message: str) -> None:
        await self.s.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(
                status=DocumentStatus.FAILED,
                error_code=code,
                error_message=message[:1000],
                finished_at=utcnow(),
            )
        )

    async def requeue(self, document_id: UUID) -> None:
        """Перед ретраем транзиентной ошибки: следующая попытка должна пройти claim."""
        await self.s.execute(
            update(Document)
            .where(Document.id == document_id, Document.status == DocumentStatus.PROCESSING)
            .values(status=DocumentStatus.QUEUED)
        )
