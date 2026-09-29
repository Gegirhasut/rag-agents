"""Ingest (ARCHITECTURE §2.2, §10): parse → фан-аут батчей эмбеддинга → финализация.

Задача parse (очередь ingest.parse) парсит файл, пишет чанки в PG и публикует по задаче на
батч (ingest.embed). Батчи независимы: ретраятся по отдельности, а документ становится done
тем батчем, чей атомарный +1 сделал batches_done == batches_total.
"""

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import timedelta
from itertools import islice
from uuid import UUID

import anyio
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agents.core import metrics
from rag_agents.core.config import Settings
from rag_agents.core.db import Database
from rag_agents.core.errors import PermanentError, TransientError
from rag_agents.core.observability import Span, Tracer
from rag_agents.core.storage import LocalFileStorage
from rag_agents.domain.agents import AgentIndexOut, AgentOut
from rag_agents.domain.documents import ChunkDraft, ChunkPayload, DocumentOut
from rag_agents.domain.enums import DocumentStatus, IngestStage, JobStatus
from rag_agents.domain.tasks import DeleteDocumentTask, EmbedBatchTask, ParseTask, PurgeAgentTask
from rag_agents.rag.chunking.structural import StructuralChunker
from rag_agents.rag.embeddings.ollama import Embedder
from rag_agents.rag.index.qdrant import QdrantChunkIndex, chunk_point_id
from rag_agents.rag.parsing.base import ParsedMeta, Section
from rag_agents.rag.parsing.registry import parser_for
from rag_agents.repositories.agents import AgentRepository
from rag_agents.repositories.chunks import ChunkRepository
from rag_agents.repositories.documents import DocumentRepository
from rag_agents.repositories.jobs import JobRepository
from rag_agents.services.progress import ProgressStore
from rag_agents.services.publisher import TaskPublisher
from rag_agents.services.trace import TraceBus

log = structlog.get_logger()

SAVE_BATCH = 500  # чанков на INSERT и heartbeat (§10.5)
MAX_CHUNKS = 20_000  # ~8 млн токенов: больше — failed: too_large
QUEUED_STALE = timedelta(minutes=2)
PROCESSING_STALE = timedelta(minutes=10)
DELETING_STALE = timedelta(minutes=5)
# Ключ «хаос-переключателя» (dev): следующий батч эмбеддинга упадёт с внутренней ошибкой →
# сообщение уйдёт в ingest.embed.dlq. Для демо replay (make chaos-embed, make dlq-replay)
CHAOS_EMBED_KEY = "chaos:ingest.embed"


def _ms(since: float) -> int:
    return int((time.monotonic() - since) * 1000)


def batch_ranges(total: int, size: int) -> list[tuple[int, int]]:
    """(ord_from, ord_to) включительно для каждого батча."""
    return [(a, min(a + size, total) - 1) for a in range(0, total, size)]


@dataclass(frozen=True)
class SweepReport:
    requeued: int = 0
    restarted: int = 0
    deletions: int = 0


class IngestService:
    def __init__(
        self,
        db: Database,
        storage: LocalFileStorage,
        chunker_factory: Callable[[], StructuralChunker],
        embedder: Embedder,
        index: QdrantChunkIndex,
        progress: ProgressStore,
        trace: TraceBus,
        settings: Settings,
        tracer: Tracer,
        publisher: TaskPublisher,
    ) -> None:
        self.db = db
        self.storage = storage
        # Фабрика, а не готовый чанкер: токенизатор (~240 МБ) нужен только worker-ingest
        self.chunker_factory = chunker_factory
        self.embedder = embedder
        self.index = index
        self.progress = progress
        self.trace = trace
        self.settings = settings
        self.tracer = tracer
        self.publisher = publisher

    def _trace_id(self, document_id: UUID, job_id: UUID) -> str:
        """Один трейс на попытку: parse и все батчи из разных процессов пишут в него."""
        return self.tracer.trace_id_for(f"ingest:{document_id}:{job_id}")

    # ------------------------------------------------------------------ parse

    async def parse(self, task: ParseTask, *, final_attempt: bool, attempt: int = 0) -> None:
        """Парсинг, чанкинг, запись чанков и публикация батчей эмбеддинга."""
        doc_id = task.document_id
        async with self.db.uow() as uow:
            claimed = await DocumentRepository(uow.session).claim(doc_id, task.job_id)
            if claimed:
                await JobRepository(uow.session).start(task.job_id)
            await uow.commit()
        if not claimed:
            log.info("ingest.parse_skip", document_id=str(doc_id), job_id=str(task.job_id))
            return
        await self.trace.emit(
            "worker.claimed",
            "rabbitmq",
            "worker",
            "worker-ingest получил задачу parse и захватил документ (UPDATE … RETURNING)",
        )
        structlog.contextvars.bind_contextvars(document_id=str(doc_id))
        try:
            n = await self._parse(task, attempt)
        except PermanentError as e:
            await self._fail(doc_id, task.job_id, e.code, e.message)
            log.warning("ingest.failed_permanent", code=e.code, error=e.message)
        except TransientError as e:
            if final_attempt:
                await self._fail(
                    doc_id, task.job_id, "transient_exhausted", "Сервис индексации недоступен"
                )
            else:
                # Следующая попытка Celery должна снова пройти claim
                async with self.db.uow() as uow:
                    await DocumentRepository(uow.session).requeue(doc_id)
                    await uow.commit()
            log.warning("ingest.transient", error=str(e), final_attempt=final_attempt)
            raise
        except Exception:
            await self._fail(doc_id, task.job_id, "internal_error", "Внутренняя ошибка обработки")
            log.exception("ingest.internal_error")
            raise
        else:
            log.info("ingest.parsed", chunks=n)
        finally:
            structlog.contextvars.unbind_contextvars("document_id")
            self.tracer.flush()

    async def _context(
        self, document_id: UUID, index_id: UUID
    ) -> tuple[DocumentOut, AgentOut, AgentIndexOut] | None:
        async with self.db.session() as s:
            doc = await DocumentRepository(s).get_for_worker(document_id)
            agents = AgentRepository(s)
            agent = await agents.get_unscoped(doc.agent_id) if doc else None
            index = await agents.get_index(agent.id, index_id) if agent else None
        if doc is None or agent is None or index is None:
            return None
        return doc, agent, index

    async def _parse(self, task: ParseTask, attempt: int) -> int:
        ctx = await self._context(task.document_id, task.index_id)
        if ctx is None:
            raise PermanentError("not_found", "Документ или агент удалены")
        doc, agent, index = ctx
        root = self.tracer.start_trace(
            "ingest",
            trace_id=self._trace_id(doc.id, task.job_id),
            user_id=str(agent.owner_id),
            session_id=f"document-{doc.id}",
            tags=[agent.name, "ingest"],
            input={"document_id": str(doc.id), "filename": doc.filename, "format": doc.format},
            metadata={
                "agent_id": str(agent.id),
                "job_id": str(task.job_id),
                "attempt": attempt,
                "collection": index.collection,
                "embedding_model": self.settings.embedding_model,
            },
        )
        with root:
            return await self._parse_pipeline(task, doc, agent, index, root)

    async def _parse_pipeline(
        self,
        task: ParseTask,
        doc: DocumentOut,
        agent: AgentOut,
        index: AgentIndexOut,
        root: Span,
    ) -> int:
        doc_id = doc.id
        timings: dict[str, int] = {}
        await self.progress.set(doc_id, IngestStage.PARSING)

        t = time.monotonic()
        with root.child("parse", metadata={"format": doc.format}):
            path = self.storage.path(doc.storage_key)
            if not await anyio.Path(path).is_file():
                # Ретрай не поможет: оригинал потерян (удалён том, ручная чистка)
                raise PermanentError("file_missing", "Файл загрузки не найден — загрузите заново")
            # Парсеры синхронные (lxml, PyMuPDF): открытие файла — в поток
            meta, sections = await anyio.to_thread.run_sync(
                parser_for(doc.format).parse, path, doc.filename
            )
        timings["parse_ms"] = _ms(t)

        async with self.db.uow() as uow:
            await ChunkRepository(uow.session).delete_for_document(agent.id, doc_id)
            await uow.commit()

        t = time.monotonic()
        with root.child(
            "chunk",
            metadata={
                "chunker": "structural",
                "target_tokens": self.settings.chunk_target_tokens,
                "max_tokens": self.settings.chunk_max_tokens,
            },
        ) as span:
            total, save_ms = await self._chunk_and_save(doc, agent, index, meta, sections)
            span.update(output={"chunks": total, "toc_source": meta.toc_source})
        timings["chunk_ms"] = _ms(t) - save_ms
        timings["save_chunks_ms"] = save_ms
        if total == 0:
            raise PermanentError("empty_document", "В документе не найдено текста")
        await self.progress.set(doc_id, IngestStage.CHUNKING, 1.0)
        await self.trace.emit(
            "ingest.chunks_saved",
            "worker",
            "postgres",
            f"Парсинг {doc.format} и чанкинг: {total} чанков записаны в PG батчами по {SAVE_BATCH}",
            agent_id=agent.id,
            title=meta.title,
            chunks=total,
        )

        ranges = batch_ranges(total, self.settings.embedding_batch_size)
        with root.child("fanout", metadata={"batches": len(ranges)}):
            await self.index.ensure_collection(index.collection, index.dim)
            # Прежняя попытка могла оставить точки с ord за пределами нового числа чанков
            await self.index.delete_document(index.collection, agent.id, doc_id)
            tasks = [
                EmbedBatchTask(
                    document_id=doc_id, job_id=task.job_id, index_id=index.id, batch_no=n
                )
                for n in range(1, len(ranges) + 1)
            ]
            async with self.db.uow() as uow:
                started = await DocumentRepository(uow.session).start_embedding(
                    doc_id,
                    task.job_id,
                    title=meta.title,
                    author=meta.author,
                    meta=meta.model_dump(),
                    chunks_total=total,
                    batches_total=len(ranges),
                )
                if not started:
                    log.info("ingest.parse_superseded", document_id=str(doc_id))
                    return total
                jobs = JobRepository(uow.session)
                await jobs.create_batches(task.job_id, ranges)
                await jobs.set_stats(task.job_id, timings)
                uow.on_commit(lambda: anyio.to_thread.run_sync(self.publisher.publish_embed, tasks))
                await uow.commit()
        await self.progress.set(doc_id, IngestStage.EMBEDDING, 0.0)
        await self.trace.emit(
            "celery.fanout",
            "worker",
            "rabbitmq",
            f"Фан-аут: {len(ranges)} задач эмбеддинга → очередь ingest.embed",
            agent_id=agent.id,
        )
        root.update(output={"chunks_total": total, "batches": len(ranges), **timings})
        return total

    async def _chunk_and_save(
        self,
        doc: DocumentOut,
        agent: AgentOut,
        index: AgentIndexOut,
        meta: ParsedMeta,
        sections: Iterator[Section],
    ) -> tuple[int, int]:
        """Чанки пишутся в PG пачками по SAVE_BATCH: книга целиком в памяти не лежит.

        Генератор парсера и чанкера синхронный (CPU): каждую пачку считаем в потоке.
        """
        chunker = self.chunker_factory()
        drafts = chunker.chunk(meta, sections)

        def next_batch() -> list[ChunkDraft]:
            return list(islice(drafts, SAVE_BATCH))

        total = save_ms = 0
        while batch := await anyio.to_thread.run_sync(next_batch):
            total += len(batch)
            if total > MAX_CHUNKS:
                raise PermanentError(
                    "too_large",
                    f"Документ больше {MAX_CHUNKS} фрагментов — разбейте файл на части",
                )
            t = time.monotonic()
            rows = [(chunk_point_id(doc.id, index.chunking_version, d.ord), d) for d in batch]
            async with self.db.uow() as uow:
                await ChunkRepository(uow.session).insert(agent.id, doc.id, rows)
                await DocumentRepository(uow.session).heartbeat(doc.id)
                await uow.commit()
            save_ms += _ms(t)
            await self.progress.set(doc.id, IngestStage.CHUNKING, 0.0)
        return total, save_ms

    # ------------------------------------------------------------------ embed

    async def embed_batch(
        self, task: EmbedBatchTask, *, final_attempt: bool, attempt: int = 0
    ) -> None:
        """Эмбеддинг и upsert одного батча; последний готовый батч финализирует документ."""
        structlog.contextvars.bind_contextvars(
            document_id=str(task.document_id), batch_no=task.batch_no
        )
        try:
            await self._embed_batch(task, attempt)
        except PermanentError as e:
            await self._fail(task.document_id, task.job_id, e.code, e.message)
            log.warning("ingest.embed_permanent", code=e.code)
        except TransientError as e:
            if final_attempt:
                await self._fail_batch(task, "transient_exhausted", "Сервис индексации недоступен")
            log.warning("ingest.embed_transient", error=str(e), final_attempt=final_attempt)
            raise
        except Exception:
            await self._fail_batch(task, "internal_error", "Внутренняя ошибка обработки")
            log.exception("ingest.embed_internal_error")
            raise
        finally:
            structlog.contextvars.unbind_contextvars("document_id", "batch_no")
            self.tracer.flush()

    async def _embed_batch(self, task: EmbedBatchTask, attempt: int) -> None:
        ctx = await self._context(task.document_id, task.index_id)
        if ctx is None:
            log.info("ingest.embed_skip_missing")
            return
        doc, agent, index = ctx
        # Задача прежней попытки или документ уже удаляют — ничего не делаем (ack)
        if doc.job_id != task.job_id or doc.status not in (
            DocumentStatus.PROCESSING,
            DocumentStatus.FAILED,
        ):
            log.info("ingest.embed_skip_stale", status=doc.status)
            return
        async with self.db.session() as s:
            batch = await JobRepository(s).get_batch(task.job_id, task.batch_no)
            if batch is None or batch[2] == JobStatus.DONE:
                log.info("ingest.embed_skip_done")
                return
            chunks = await ChunkRepository(s).range(agent.id, doc.id, batch[0], batch[1])
        if not chunks:
            raise PermanentError("chunks_missing", "Фрагменты документа не найдены")
        if self.settings.app_env != "prod" and await self.progress.redis.getdel(CHAOS_EMBED_KEY):
            raise RuntimeError("chaos: искусственный сбой батча эмбеддинга")

        root = self.tracer.start_trace(
            "embed_batch",
            trace_id=self._trace_id(doc.id, task.job_id),
            user_id=str(agent.owner_id),
            session_id=f"document-{doc.id}",
            tags=[agent.name, "ingest"],
            metadata={
                "batch": f"{task.batch_no}/{doc.batches_total}",
                "texts": len(chunks),
                "attempt": attempt,
            },
        )
        with root:
            t = time.monotonic()
            with root.child("embed", as_type="embedding", model=self.settings.embedding_model):
                vectors = await self.embedder.embed([c.embed_text for c in chunks])
            embed_ms = _ms(t)
            metrics.EMBED_BATCH_SECONDS.observe(embed_ms / 1000)
            points = [
                (
                    c.id,
                    vec,
                    ChunkPayload(
                        agent_id=str(agent.id),
                        document_id=str(doc.id),
                        chunk_id=str(c.id),
                        ord=c.ord,
                        book_title=doc.title,
                        author=doc.author,
                        section_path=c.section_path,
                        chapter_title=c.chapter_title,
                        page_from=c.page_from,
                        page_to=c.page_to,
                        text=c.text,
                    ),
                )
                for c, vec in zip(chunks, vectors, strict=True)
            ]
            t = time.monotonic()
            with root.child("upsert", metadata={"points": len(points)}):
                await self.index.upsert(index.collection, agent.id, points)
            upsert_ms = _ms(t)
            await self.trace.emit(
                "ingest.embedded",
                "ollama",
                "qdrant",
                f"Батч {task.batch_no}/{doc.batches_total}: {len(points)} векторов → upsert "
                f"({embed_ms} + {upsert_ms} мс)",
                agent_id=agent.id,
                batch=task.batch_no,
                batches=doc.batches_total,
            )

            # Документ удалили, пока шёл батч: убираем только что записанные точки
            async with self.db.session() as s:
                current = await DocumentRepository(s).get_for_worker(doc.id)
            if current is None or current.status == DocumentStatus.DELETING:
                await self.index.delete_document(index.collection, agent.id, doc.id)
                return

            await self._complete(task, agent, root, embed_ms, upsert_ms)

    async def _complete(
        self, task: EmbedBatchTask, agent: AgentOut, root: Span, embed_ms: int, upsert_ms: int
    ) -> None:
        finished = False
        progress: tuple[int, int] | None = None
        async with self.db.uow() as uow:
            jobs = JobRepository(uow.session)
            docs = DocumentRepository(uow.session)
            if await jobs.complete_batch(
                task.job_id, task.batch_no, embed_ms=embed_ms, upsert_ms=upsert_ms
            ):
                progress = await docs.increment_batches(task.document_id, task.job_id)
                if progress is not None and progress[0] >= progress[1]:
                    # Этот батч — последний: финализирует именно он (решает атомарный +1)
                    with root.child("finalize"):
                        stats = await jobs.get_stats(task.job_id)
                        emb, ups = await jobs.batch_timings(task.job_id)
                        await docs.mark_done(
                            task.document_id, {**stats, "embed_ms": emb, "upsert_ms": ups}
                        )
                        await jobs.finish(task.job_id, JobStatus.DONE)
                        await AgentRepository(uow.session).bump_corpus_version(agent.id)
                    finished = True
            await uow.commit()
        if finished:
            await self.progress.set(task.document_id, IngestStage.FINALIZING, 1.0)
            await self.trace.emit(
                "ingest.done",
                "worker",
                "postgres",
                "Последний батч: status=done, corpus_version+1 — документ доступен для поиска",
                agent_id=agent.id,
            )
            log.info("ingest.done")
            await self._observe_finished(task.document_id)
        elif progress is not None:
            await self.progress.set(
                task.document_id, IngestStage.EMBEDDING, progress[0] / max(progress[1], 1)
            )

    async def _fail_batch(self, task: EmbedBatchTask, code: str, message: str) -> None:
        async with self.db.uow() as uow:
            await JobRepository(uow.session).fail_batch(task.job_id, task.batch_no)
            await uow.commit()
        await self._fail(task.document_id, task.job_id, code, message)

    async def _fail(self, doc_id: UUID, job_id: UUID, code: str, message: str) -> None:
        async with self.db.uow() as uow:
            await DocumentRepository(uow.session).mark_failed(doc_id, code, message)
            await JobRepository(uow.session).finish(job_id, JobStatus.FAILED, code)
            await uow.commit()
        await self.trace.emit("ingest.failed", "worker", "postgres", f"failed: {code}")
        await self._observe_finished(doc_id)

    async def _observe_finished(self, doc_id: UUID) -> None:
        """Prometheus: документ дошёл до done/failed (формат и длительность — из PG)."""
        async with self.db.session() as s:
            doc = await DocumentRepository(s).get_for_worker(doc_id)
        if doc is None:
            return
        fmt = doc.format.value
        metrics.INGEST_DOCUMENTS.labels(doc.status.value, fmt).inc()
        if doc.status == DocumentStatus.DONE:
            metrics.INGEST_CHUNKS.inc(doc.chunks_total or 0)
            if doc.duration_s is not None:
                metrics.INGEST_SECONDS.labels(fmt).observe(doc.duration_s)

    # ------------------------------------------------------------------ maintenance

    async def delete_document(self, task: DeleteDocumentTask) -> None:
        """Qdrant → PG (чанки, задания каскадом) → файл. Повтор безопасен."""
        async with self.db.session() as s:
            doc = await DocumentRepository(s).get_for_worker(task.document_id)
            indexes = await AgentRepository(s).list_indexes(task.agent_id)
        if doc is None or doc.agent_id != task.agent_id:
            return
        if doc.status != DocumentStatus.DELETING:
            log.info("document.delete_skip", status=doc.status)
            return
        for idx in indexes:
            await self.index.delete_document(idx.collection, task.agent_id, doc.id)
        async with self.db.uow() as uow:
            await DocumentRepository(uow.session).delete_row(doc.id)
            await AgentRepository(uow.session).bump_corpus_version(task.agent_id)
            await uow.commit()
        await self.storage.delete_document_dir(doc.storage_key)
        await self.progress.clear(doc.id)
        log.info("document.deleted", document_id=str(doc.id))

    async def purge_agent(self, task: PurgeAgentTask) -> None:
        """Данные мягко удалённого агента: точки во всех коллекциях, документы, файлы."""
        async with self.db.session() as s:
            agent = await AgentRepository(s).get_any(task.agent_id)
            indexes = await AgentRepository(s).list_indexes(task.agent_id)
            docs = await DocumentRepository(s).list_ids_for_agent(task.agent_id)
        if agent is None:
            return
        if agent.deleted_at is None:
            log.warning("agent.purge_skip_alive", agent_id=str(task.agent_id))
            return
        for idx in indexes:
            await self.index.delete_agent(idx.collection, task.agent_id)
        async with self.db.uow() as uow:
            for d in docs:
                await DocumentRepository(uow.session).delete_row(d.id)
            await uow.commit()
        await self.storage.delete_agent_dir(task.agent_id)
        log.info("agent.purged", agent_id=str(task.agent_id), documents=len(docs))

    async def sweep(self) -> SweepReport:
        """Outbox-lite (§10.4): документы, застрявшие из-за потерянных сообщений или
        упавших воркеров, переотправляются. Задачи идемпотентны, дубли безопасны."""
        requeued = restarted = deletions = 0
        async with self.db.uow() as uow:
            docs = DocumentRepository(uow.session)
            parse: list[ParseTask] = []
            delete: list[DeleteDocumentTask] = []
            for d in await docs.stale(DocumentStatus.QUEUED, QUEUED_STALE):
                index_id = await self._index_for(uow.session, d)
                if index_id is None:
                    continue
                job_id = d.job_id or await docs.new_job(d.id, index_id)
                await docs.touch(d.id)
                parse.append(ParseTask(document_id=d.id, job_id=job_id, index_id=index_id))
                requeued += 1
            for d in await docs.stale(DocumentStatus.PROCESSING, PROCESSING_STALE):
                index_id = await self._index_for(uow.session, d)
                if index_id is None:
                    continue
                # Воркер умер посреди работы: новая попытка целиком, старые батчи станут no-op
                job_id = await docs.new_job(d.id, index_id)
                await docs.requeue_stale(d.id)
                parse.append(ParseTask(document_id=d.id, job_id=job_id, index_id=index_id))
                restarted += 1
            for d in await docs.stale(DocumentStatus.DELETING, DELETING_STALE):
                await docs.touch(d.id)
                delete.append(DeleteDocumentTask(document_id=d.id, agent_id=d.agent_id))
                deletions += 1

            def publish() -> None:
                for p in parse:
                    self.publisher.publish_parse(p)
                for t in delete:
                    self.publisher.publish_delete_document(t)

            uow.on_commit(lambda: anyio.to_thread.run_sync(publish))
            await uow.commit()
        report = SweepReport(requeued=requeued, restarted=restarted, deletions=deletions)
        if requeued or restarted or deletions:
            log.info("ingest.sweep", requeued=requeued, restarted=restarted, deletions=deletions)
        return report

    @staticmethod
    async def _index_for(session: AsyncSession, doc: DocumentOut) -> UUID | None:
        agent = await AgentRepository(session).get_unscoped(doc.agent_id)
        return agent.active_index_id if agent else None
