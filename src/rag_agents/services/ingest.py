import time
from uuid import UUID

import structlog

from rag_agents.core.config import Settings
from rag_agents.core.db import Database
from rag_agents.core.errors import PermanentError, TransientError
from rag_agents.core.observability import Span, Tracer
from rag_agents.core.storage import LocalFileStorage
from rag_agents.domain.agents import AgentIndexOut, AgentOut
from rag_agents.domain.documents import ChunkDraft, ChunkPayload, DocumentOut
from rag_agents.domain.enums import IngestStage
from rag_agents.domain.tasks import IngestDocumentTask
from rag_agents.rag.chunking.naive import NaiveChunker
from rag_agents.rag.embeddings.ollama import Embedder
from rag_agents.rag.index.qdrant import QdrantChunkIndex, chunk_point_id
from rag_agents.rag.parsing.base import ParsedMeta
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
        tracer: Tracer,
    ) -> None:
        self.tracer = tracer
        self.db = db
        self.storage = storage
        self.chunker = chunker
        self.embedder = embedder
        self.index = index
        self.progress = progress
        self.trace = trace
        self.settings = settings

    async def ingest(
        self, task: IngestDocumentTask, *, final_attempt: bool, attempt: int = 0
    ) -> None:
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
            chunks_total = await self._run(task, attempt)
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
            # Воркер живёт долго, но процесс может быть переработан (max_tasks_per_child):
            # отправляем трейс документа сразу, не дожидаясь фонового экспорта
            self.tracer.flush()

    async def _run(self, task: IngestDocumentTask, attempt: int) -> int:
        doc_id = task.document_id
        async with self.db.session() as s:
            doc = await DocumentRepository(s).get_for_worker(doc_id)
            agent = await AgentRepository(s).get_unscoped(doc.agent_id) if doc else None
            index = await AgentRepository(s).get_index(agent.id, task.index_id) if agent else None
        if doc is None or agent is None or index is None:
            raise PermanentError("not_found", "Документ или агент удалены")

        # Трейс на попытку ingest: session = документ (ретраи видны рядом), тексты не пишем
        root = self.tracer.start_trace(
            "ingest",
            trace_id=self.tracer.trace_id_for(f"ingest:{doc_id}:{attempt}"),
            user_id=str(agent.owner_id),
            session_id=f"document-{doc_id}",
            tags=[agent.name, "ingest"],
            input={"document_id": str(doc_id), "filename": doc.filename, "format": "txt"},
            metadata={
                "agent_id": str(agent.id),
                "document_id": str(doc_id),
                "attempt": attempt,
                "collection": index.collection,
                "embedding_model": self.settings.embedding_model,
            },
        )
        with root:
            return await self._pipeline(task, doc, agent, index, root)

    async def _pipeline(
        self,
        task: IngestDocumentTask,
        doc: DocumentOut,
        agent: AgentOut,
        index: AgentIndexOut,
        root: Span,
    ) -> int:
        """parse → chunk → save_chunks → embed_upsert (батчи) → finalize, каждый шаг — span."""
        doc_id = task.document_id
        started = time.monotonic()
        meta, drafts = await self._parse_and_chunk(doc, agent, root)
        ids = [chunk_point_id(doc_id, index.chunking_version, d.ord) for d in drafts]
        with root.child("save_chunks", metadata={"rows": len(drafts)}):
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

        n_total, embed_ms, upsert_ms = await self._embed_and_upsert(
            doc_id, agent, index, meta, drafts, ids, root
        )

        await self.progress.set(doc_id, IngestStage.FINALIZING)
        with root.child("finalize"):
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
        root.update(
            output={
                "chunks_total": len(drafts),
                "batches": n_total,
                "embed_ms": embed_ms,
                "upsert_ms": upsert_ms,
                "duration_ms": int((time.monotonic() - started) * 1000),
            }
        )
        return len(drafts)

    async def _parse_and_chunk(
        self, doc: DocumentOut, agent: AgentOut, root: Span
    ) -> tuple[ParsedMeta, list[ChunkDraft]]:
        doc_id = doc.id
        await self.progress.set(doc_id, IngestStage.PARSING)
        t = time.monotonic()
        with root.child("parse", metadata={"format": "txt"}) as span:
            meta, sections = TxtParser().parse(self.storage.path(doc.storage_key), doc.filename)
            section_list = list(sections)
            span.update(output={"sections": len(section_list), "has_title": bool(meta.title)})
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
        with root.child(
            "chunk",
            metadata={
                "chunker": "naive",
                "target_tokens": self.settings.chunk_target_tokens,
                "max_tokens": self.settings.chunk_max_tokens,
            },
        ) as span:
            drafts = self.chunker.chunk(meta, section_list)
            if not drafts:
                raise PermanentError("empty_document", "В документе не найдено текста")
            tokens = [d.token_count for d in drafts]
            span.update(
                output={
                    "chunks": len(drafts),
                    "avg_tokens": sum(tokens) // len(tokens),
                    "max_tokens": max(tokens),
                }
            )
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
        return meta, drafts

    async def _embed_and_upsert(
        self,
        doc_id: UUID,
        agent: AgentOut,
        index: AgentIndexOut,
        meta: ParsedMeta,
        drafts: list[ChunkDraft],
        ids: list[UUID],
        root: Span,
    ) -> tuple[int, int, int]:
        """Батчи: эмбеддинг (Ollama) → upsert (Qdrant). Возвращает (батчей, мс embed, мс upsert)."""
        batch = self.settings.embedding_batch_size
        n_total = -(-len(drafts) // batch)
        embed_ms = upsert_ms = 0
        # Эмбеддинг и upsert чередуются по батчам: общий родитель, под ним embed_batch и
        # upsert_batch по очереди — в UI видна длительность каждого шага каждого батча
        with root.child(
            "embed_upsert",
            metadata={"batch_size": batch, "batches": n_total, "collection": index.collection},
        ) as stage:
            await self.index.ensure_collection(index.collection, index.dim)
            # Повторный ingest мог дать меньше чанков — старые точки документа убираем
            await self.index.delete_document(index.collection, agent.id, doc_id)
            for start in range(0, len(drafts), batch):
                part = drafts[start : start + batch]
                n_batch = start // batch + 1
                t = time.monotonic()
                with stage.child(
                    # Имя стабильное (агрегаты Langfuse группируют по имени), номер — в metadata
                    "embed_batch",
                    as_type="embedding",
                    model=self.settings.embedding_model,
                    metadata={"batch": f"{n_batch}/{n_total}", "texts": len(part)},
                ):
                    vectors = await self.embedder.embed([d.embed_text for d in part])
                batch_embed_ms = int((time.monotonic() - t) * 1000)
                embed_ms += batch_embed_ms
                await self.trace.emit(
                    "ingest.embedded",
                    "ollama",
                    "worker",
                    f"Батч {n_batch}/{n_total}: текстов {len(part)} → "
                    f"векторы {len(vectors[0])}-d ({batch_embed_ms} мс)",
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
                t = time.monotonic()
                with stage.child(
                    "upsert_batch",
                    metadata={"batch": f"{n_batch}/{n_total}", "points": len(points)},
                ):
                    await self.index.upsert(index.collection, agent.id, points)
                upsert_ms += int((time.monotonic() - t) * 1000)
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
            stage.update(
                output={"points": len(drafts), "embed_ms": embed_ms, "upsert_ms": upsert_ms}
            )
        return n_total, embed_ms, upsert_ms

    async def _fail(self, doc_id: UUID, code: str, message: str) -> None:
        async with self.db.uow() as uow:
            await DocumentRepository(uow.session).mark_failed(doc_id, code, message)
            await uow.commit()
