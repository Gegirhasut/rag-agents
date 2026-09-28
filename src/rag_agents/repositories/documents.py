from datetime import timedelta
from uuid import UUID

from sqlalchemy import delete, func, or_, select, type_coerce, update
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agents.core.ids import utcnow, uuid7
from rag_agents.domain.documents import DocumentOut
from rag_agents.domain.enums import DocumentStatus, IngestStage
from rag_agents.models.entities import Document, IngestJob

STALE_AFTER = timedelta(minutes=10)
# failed с этими кодами можно захватить повторно (replay из DLQ): сбой был не в данных
RETRIABLE_CODES = ("transient_exhausted", "internal_error")


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

    async def claim(self, document_id: UUID, job_id: UUID) -> bool:
        """Условный UPDATE вместо SELECT-then-UPDATE: повторная доставка задачи — no-op.

        Захватить можно queued, «протухший» processing (воркер умер) и failed после
        инфраструктурной ошибки — последнее нужно для replay из DLQ. Только текущая попытка.
        """
        now = utcnow()
        result = await self.s.execute(
            update(Document)
            .where(
                Document.id == document_id,
                Document.job_id == job_id,
                or_(
                    Document.status == DocumentStatus.QUEUED,
                    (Document.status == DocumentStatus.PROCESSING)
                    & (Document.heartbeat_at < now - STALE_AFTER),
                    (Document.status == DocumentStatus.FAILED)
                    & Document.error_code.in_(RETRIABLE_CODES),
                ),
            )
            .values(
                status=DocumentStatus.PROCESSING,
                stage=IngestStage.PARSING.value,
                heartbeat_at=now,
                started_at=now,
                finished_at=None,
                error_code=None,
                error_message=None,
                batches_total=None,
                batches_done=0,
            )
            .returning(Document.id)
        )
        return result.scalar_one_or_none() is not None

    async def new_job(self, document_id: UUID, index_id: UUID) -> UUID:
        """Новая попытка обработки: задачи прежних попыток после этого — no-op."""
        job_id = uuid7()
        self.s.add(IngestJob(id=job_id, document_id=document_id, index_id=index_id))
        await self.s.flush()
        await self.s.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(job_id=job_id, heartbeat_at=utcnow())
        )
        return job_id

    async def start_embedding(
        self,
        document_id: UUID,
        job_id: UUID,
        *,
        title: str | None,
        author: str | None,
        meta: dict[str, object],
        chunks_total: int,
        batches_total: int,
    ) -> bool:
        """Конец парсинга. False — документ тем временем удаляют или перезапустили."""
        found = await self.s.scalar(
            update(Document)
            .where(
                Document.id == document_id,
                Document.job_id == job_id,
                Document.status == DocumentStatus.PROCESSING,
            )
            .values(
                stage=IngestStage.EMBEDDING.value,
                title=title,
                author=author,
                meta=meta,
                chunks_total=chunks_total,
                batches_total=batches_total,
                batches_done=0,
                heartbeat_at=utcnow(),
            )
            .returning(Document.id)
        )
        return found is not None

    async def increment_batches(self, document_id: UUID, job_id: UUID) -> tuple[int, int] | None:
        """+1 готовый батч (атомарно). Возвращает (готово, всего) для решения о финализации."""
        row = (
            await self.s.execute(
                update(Document)
                .where(Document.id == document_id, Document.job_id == job_id)
                .values(batches_done=Document.batches_done + 1, heartbeat_at=utcnow())
                .returning(Document.batches_done, Document.batches_total)
            )
        ).first()
        return (row[0], row[1] or 0) if row else None

    async def mark_deleting(self, agent_id: UUID, document_id: UUID) -> DocumentOut | None:
        doc = await self.s.scalar(
            update(Document)
            .where(
                Document.id == document_id,
                Document.agent_id == agent_id,
                Document.status != DocumentStatus.DELETING,
            )
            .values(status=DocumentStatus.DELETING, heartbeat_at=utcnow())
            .returning(Document)
        )
        return DocumentOut.model_validate(doc) if doc else None

    async def delete_row(self, document_id: UUID) -> None:
        await self.s.execute(delete(Document).where(Document.id == document_id))

    async def list_ids_for_agent(self, agent_id: UUID) -> list[DocumentOut]:
        """Для очистки агента: все документы, включая deleting."""
        rows = await self.s.scalars(select(Document).where(Document.agent_id == agent_id))
        return [DocumentOut.model_validate(d) for d in rows]

    async def stale(self, status: DocumentStatus, older_than: timedelta) -> list[DocumentOut]:
        """Для sweeper: документы, которые слишком долго не двигаются."""
        cutoff = utcnow() - older_than
        rows = await self.s.scalars(
            select(Document)
            .where(
                Document.status == status,
                func.coalesce(Document.heartbeat_at, Document.created_at) < cutoff,
            )
            .limit(100)
        )
        return [DocumentOut.model_validate(d) for d in rows]

    async def touch(self, document_id: UUID) -> None:
        await self.heartbeat(document_id)

    async def requeue_stale(self, document_id: UUID) -> None:
        await self.s.execute(
            update(Document)
            .where(Document.id == document_id, Document.status == DocumentStatus.PROCESSING)
            .values(status=DocumentStatus.QUEUED, stage=None)
        )

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
        """timings (мс по шагам) дописываются в meta.timings, остальной meta не трогаем.

        Ошибка сбрасывается: документ мог быть failed, пока батч лежал в DLQ (replay).
        """
        patch = type_coerce({"timings": timings or {}}, JSONB)
        await self.s.execute(
            update(Document)
            .where(Document.id == document_id, Document.status != DocumentStatus.DELETING)
            .values(
                status=DocumentStatus.DONE,
                stage=None,
                error_code=None,
                error_message=None,
                finished_at=utcnow(),
                meta=Document.meta.op("||")(patch),
            )
        )

    async def mark_failed(self, document_id: UUID, code: str, message: str) -> None:
        await self.s.execute(
            update(Document)
            .where(Document.id == document_id, Document.status != DocumentStatus.DELETING)
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

    async def reset_failed(self, agent_id: UUID, document_id: UUID) -> DocumentOut | None:
        """failed → queued для повторной обработки. Из других статусов — None (не повторяем)."""
        doc = await self.s.scalar(
            update(Document)
            .where(
                Document.id == document_id,
                Document.agent_id == agent_id,
                Document.status == DocumentStatus.FAILED,
            )
            .values(
                status=DocumentStatus.QUEUED,
                stage=None,
                error_code=None,
                error_message=None,
                started_at=None,
                finished_at=None,
                batches_done=0,
            )
            .returning(Document)
        )
        return DocumentOut.model_validate(doc) if doc else None
