import time
from uuid import UUID

import structlog

from rag_agents.core.config import Settings
from rag_agents.core.db import Database
from rag_agents.core.errors import PermanentError, TransientError
from rag_agents.core.storage import LocalFileStorage
from rag_agents.domain.documents import ChunkPayload
from rag_agents.domain.enums import IngestStage
from rag_agents.domain.tasks import IngestDocumentTask
from rag_agents.rag.chunking.naive import NaiveChunker
from rag_agents.rag.embeddings.ollama import Embedder
from rag_agents.rag.index.qdrant import QdrantChunkIndex, chunk_point_id
from rag_agents.rag.parsing.txt import TxtParser
from rag_agents.repositories.agents import AgentRepository
from rag_agents.repositories.chunks import ChunkRepository
from rag_agents.repositories.documents import DocumentRepository
from rag_agents.services.progress import ProgressStore
from rag_agents.services.trace import TraceBus

log = structlog.get_logger()


class IngestService:
    """Итерация 1: одна задача на документ (без фан-аута батчей — это итерация 3)."""

    def __init__(
        self,
        db: Database,
        storage: LocalFileStorage,
        chunker: NaiveChunker,
        embedder: Embedder,
        index: QdrantChunkIndex,
        progress: ProgressStore,
        trace: TraceBus,
        settings: Settings,
    ) -> None:
        self.db = db
        self.storage = storage
        self.chunker = chunker
        self.embedder = embedder
        self.index = index
        self.progress = progress
        self.trace = trace
        self.settings = settings

    async def ingest(self, task: IngestDocumentTask, *, final_attempt: bool) -> None:
        doc_id = task.document_id
        async with self.db.uow() as uow:
            claimed = await DocumentRepository(uow.session).claim(doc_id)
            await uow.commit()
        if not claimed:
            log.info("ingest.skip_not_claimable", document_id=str(doc_id))
            return
        await self.trace.emit(
            "worker.claimed",
            "rabbitmq",
            "worker",
            "Celery-воркер получил задачу и захватил документ (UPDATE … RETURNING)",
        )
        structlog.contextvars.bind_contextvars(document_id=str(doc_id))
        started = time.monotonic()
        try:
            chunks_total = await self._run(task)
        except PermanentError as e:
            await self._fail(doc_id, e.code, e.message)
            await self.trace.emit(
                "ingest.failed", "worker", "postgres", f"Ошибка без ретрая: {e.code} → failed"
            )
            log.warning("ingest.failed_permanent", code=e.code, error=e.message)
        except TransientError as e:
            if final_attempt:
                await self._fail(doc_id, "transient_exhausted", "Сервис индексации недоступен")
            else:
                async with self.db.uow() as uow:
                    await DocumentRepository(uow.session).requeue(doc_id)
                    await uow.commit()
            log.warning("ingest.transient", error=str(e), final_attempt=final_attempt)
            raise
        except Exception:
            await self._fail(doc_id, "internal_error", "Внутренняя ошибка обработки")
            log.exception("ingest.internal_error")
            raise
        else:
            log.info(
                "ingest.done",
                chunks=chunks_total,
                duration_s=round(time.monotonic() - started, 1),
            )
        finally:
            structlog.contextvars.unbind_contextvars("document_id")

    async def _run(self, task: IngestDocumentTask) -> int:
        doc_id = task.document_id
        async with self.db.session() as s:
            doc = await DocumentRepository(s).get_for_worker(doc_id)
            agent = await AgentRepository(s).get_unscoped(doc.agent_id) if doc else None
            index = await AgentRepository(s).get_index(agent.id, task.index_id) if agent else None
        if doc is None or agent is None or index is None:
            raise PermanentError("not_found", "Документ или агент удалены")

        await self.progress.set(doc_id, IngestStage.PARSING)
        t = time.monotonic()
        meta, sections = TxtParser().parse(self.storage.path(doc.storage_key), doc.filename)
        section_list = list(sections)
        await self.trace.emit(
            "ingest.parsed",
            "uploads",
            "worker",
            f"Парсинг txt: кодировка, главы → секций: {len(section_list)} "
            f"({int((time.monotonic() - t) * 1000)} мс)",
            agent_id=agent.id,
            title=meta.title,
        )

        await self.progress.set(doc_id, IngestStage.CHUNKING)
        t = time.monotonic()
        drafts = self.chunker.chunk(meta, section_list)
        if not drafts:
            raise PermanentError("empty_document", "В документе не найдено текста")
        tokens = [d.token_count for d in drafts]
        await self.trace.emit(
            "ingest.chunked",
            "worker",
            None,
            f"Чанкинг токенизатором bge-m3: чанков {len(drafts)}, "
            f"~{sum(tokens) // len(tokens)} токенов в среднем "
            f"({int((time.monotonic() - t) * 1000)} мс)",
            agent_id=agent.id,
            chunks=len(drafts),
        )
        ids = [chunk_point_id(doc_id, index.chunking_version, d.ord) for d in drafts]
        async with self.db.uow() as uow:
            await ChunkRepository(uow.session).replace_for_document(
                agent.id, doc_id, list(zip(ids, drafts, strict=True))
            )
            await DocumentRepository(uow.session).set_stage(
                doc_id,
                IngestStage.EMBEDDING,
                title=meta.title,
                author=meta.author,
                meta=meta.model_dump(),
                chunks_total=len(drafts),
            )
            await uow.commit()
        await self.trace.emit(
            "ingest.chunks_saved",
            "worker",
            "postgres",
            f"INSERT chunks ×{len(drafts)} (тексты — источник правды для переиндексации)",
            agent_id=agent.id,
        )

        await self.index.ensure_collection(index.collection, index.dim)
        # Повторный ingest мог дать меньше чанков — старые точки документа убираем
        await self.index.delete_document(index.collection, agent.id, doc_id)
        batch = self.settings.embedding_batch_size
        for start in range(0, len(drafts), batch):
            part = drafts[start : start + batch]
            n_batch, n_total = start // batch + 1, -(-len(drafts) // batch)
            t = time.monotonic()
            vectors = await self.embedder.embed([d.embed_text for d in part])
            await self.trace.emit(
                "ingest.embedded",
                "ollama",
                "worker",
                f"Батч {n_batch}/{n_total}: текстов {len(part)} → векторы {len(vectors[0])}-d "
                f"({int((time.monotonic() - t) * 1000)} мс)",
                agent_id=agent.id,
                batch=n_batch,
                batches=n_total,
            )
            points = [
                (
                    ids[d.ord],
                    vec,
                    ChunkPayload(
                        agent_id=str(agent.id),
                        document_id=str(doc_id),
                        chunk_id=str(ids[d.ord]),
                        ord=d.ord,
                        book_title=meta.title,
                        author=meta.author,
                        section_path=d.section_path,
                        chapter_title=d.chapter_title,
                        text=d.text,
                    ),
                )
                for d, vec in zip(part, vectors, strict=True)
            ]
            await self.index.upsert(index.collection, agent.id, points)
            await self.trace.emit(
                "ingest.upserted",
                "worker",
                "qdrant",
                f"UPSERT точек: {len(points)} (uuid5-id, payload с agent_id)",
                agent_id=agent.id,
            )
            done = start + len(part)
            await self.progress.set(doc_id, IngestStage.EMBEDDING, done / len(drafts))
            await self.trace.emit(
                "ingest.progress",
                "worker",
                "redis",
                f"HSET ingest:{{id}} progress={int(20 + 78 * done / len(drafts))}% "
                "(его опрашивает страница агента)",
                agent_id=agent.id,
            )
            async with self.db.uow() as uow:
                await DocumentRepository(uow.session).heartbeat(doc_id)
                await uow.commit()

        await self.progress.set(doc_id, IngestStage.FINALIZING)
        async with self.db.uow() as uow:
            await DocumentRepository(uow.session).mark_done(doc_id)
            await AgentRepository(uow.session).bump_corpus_version(agent.id)
            await uow.commit()
        await self.trace.emit(
            "ingest.done",
            "worker",
            "postgres",
            "status=done, corpus_version+1 — документ доступен для поиска",
            agent_id=agent.id,
        )
        await self.progress.set(doc_id, IngestStage.FINALIZING, 1.0)
        return len(drafts)

    async def _fail(self, doc_id: UUID, code: str, message: str) -> None:
        async with self.db.uow() as uow:
            await DocumentRepository(uow.session).mark_failed(doc_id, code, message)
            await uow.commit()
