"""Ingest целиком на реальных PG, Qdrant и Redis: parse → фан-аут батчей → done.

Очередь заменена RecordingPublisher: stack.drain() исполняет задачи, как воркеры. Так
проверяются идемпотентность повторной доставки, DLQ-replay, удаление и sweeper.
"""

from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import update

from rag_agents.core.ids import utcnow
from rag_agents.domain.agents import AgentCreate, AgentOut
from rag_agents.domain.auth import UserOut
from rag_agents.domain.documents import DocumentOut
from rag_agents.domain.enums import DocumentStatus
from rag_agents.models.entities import Document
from rag_agents.rag.index.qdrant import collection_name
from rag_agents.services.errors import ValidationError
from rag_agents.services.ingest import CHAOS_EMBED_KEY
from tests.integration.conftest import Stack

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).parents[1] / "fixtures"
BOOKS = [
    "chapters_cp1251.txt",
    "tolstoy.fb2",
    "tolstoy.epub",
    "tolstoy.pdf",
    "tolstoy.docx",
]


async def _bytes(data: bytes) -> AsyncIterator[bytes]:
    yield data


async def _setup(stack: Stack, name: str = "Ingest") -> tuple[UserOut, AgentOut]:
    user = await stack.user()
    agent = await stack.container.agents.create(user.id, AgentCreate(name=name))
    return user, agent


async def _upload(stack: Stack, user: UserOut, agent: AgentOut, fixture: str) -> DocumentOut:
    data = (FIXTURES / fixture).read_bytes()
    doc, created = await stack.container.documents.upload(user.id, agent.id, fixture, _bytes(data))
    assert created
    return doc


async def _doc(stack: Stack, user: UserOut, agent: AgentOut, doc_id: UUID) -> DocumentOut:
    return await stack.container.documents.get(user.id, agent.id, doc_id)


async def _points(stack: Stack, agent: AgentOut) -> int:
    st = stack.container.settings
    name = collection_name(st.qdrant_collection_prefix, st.embedding_model, st.embedding_dim)
    return await stack.container.query.index.count(name, agent.id)


async def test_every_format_is_ingested_with_structure(stack: Stack) -> None:
    await stack.drain()
    user, agent = await _setup(stack)
    docs = [await _upload(stack, user, agent, f) for f in BOOKS]
    await stack.drain()

    total = 0
    for d in docs:
        doc, toc, chunks, n = await stack.container.documents.detail(user.id, agent.id, d.id)
        assert doc.status == DocumentStatus.DONE, (d.filename, doc.error_message)
        assert doc.chunks_total == n > 0
        assert doc.batches_done == doc.batches_total
        assert doc.meta["toc_source"] in ("native", "heuristic")
        assert {"parse_ms", "embed_ms", "upsert_ms"} <= set(doc.meta["timings"])
        assert len(toc) >= 2, d.filename  # главы видны в оглавлении
        assert chunks[0].embed_text.startswith(doc.meta.get("author") or "")
        total += n
    assert await _points(stack, agent) == total
    pdf = next(d for d in docs if d.filename == "tolstoy.pdf")
    _, _, pdf_chunks, _ = await stack.container.documents.detail(user.id, agent.id, pdf.id)
    assert pdf_chunks[0].page_from == 1


async def test_redelivered_tasks_do_not_duplicate_work(stack: Stack) -> None:
    await stack.drain()
    user, agent = await _setup(stack)
    doc = await _upload(stack, user, agent, "tolstoy.fb2")
    [parse] = stack.publisher.tasks
    stack.publisher.tasks.clear()
    ingest = stack.container.ingest

    await ingest.parse(parse, final_attempt=False)
    await ingest.parse(parse, final_attempt=False)  # повторная доставка: claim не пройдёт
    batches = list(stack.publisher.embeds)
    stack.publisher.embeds.clear()
    assert len(batches) >= 2
    for b in batches:
        await ingest.embed_batch(b, final_attempt=False)
        await ingest.embed_batch(b, final_attempt=False)  # дубль — no-op

    final = await _doc(stack, user, agent, doc.id)
    assert final.status == DocumentStatus.DONE
    assert final.batches_done == final.batches_total == len(batches)
    assert await _points(stack, agent) == final.chunks_total
    after = await stack.container.agents.get(user.id, agent.id)
    assert after.corpus_version == agent.corpus_version + 1  # финализация ровно одна


async def test_internal_error_in_batch_then_replay_finishes_document(stack: Stack) -> None:
    await stack.drain()
    user, agent = await _setup(stack)
    doc = await _upload(stack, user, agent, "tolstoy.docx")
    await stack.container.ingest.parse(stack.publisher.tasks.pop(), final_attempt=False)
    batches = list(stack.publisher.embeds)
    stack.publisher.embeds.clear()

    await stack.container.redis.set(CHAOS_EMBED_KEY, "1")
    with pytest.raises(RuntimeError, match="chaos"):  # Celery → reject → ingest.embed.dlq
        await stack.container.ingest.embed_batch(batches[0], final_attempt=False)
    for b in batches[1:]:
        await stack.container.ingest.embed_batch(b, final_attempt=False)
    failed = await _doc(stack, user, agent, doc.id)
    assert (failed.status, failed.error_code) == (DocumentStatus.FAILED, "internal_error")
    assert failed.batches_done == len(batches) - 1

    await stack.container.ingest.embed_batch(batches[0], final_attempt=False)  # replay из DLQ
    done = await _doc(stack, user, agent, doc.id)
    assert (done.status, done.error_code) == (DocumentStatus.DONE, None)
    assert await _points(stack, agent) == done.chunks_total


async def test_transient_exhausted_marks_failed_and_retry_restarts(stack: Stack) -> None:
    await stack.drain()
    user, agent = await _setup(stack)
    doc = await _upload(stack, user, agent, "tolstoy.epub")
    await stack.container.ingest.parse(stack.publisher.tasks.pop(), final_attempt=False)
    old_batches = list(stack.publisher.embeds)
    stack.publisher.embeds.clear()
    async with stack.container.db.uow() as uow:
        await uow.session.execute(
            update(Document)
            .where(Document.id == doc.id)
            .values(status=DocumentStatus.FAILED, error_code="broken", error_message="x")
        )
        await uow.commit()

    await stack.container.documents.retry(user.id, agent.id, doc.id)  # новая попытка (job)
    for b in old_batches:  # батчи старой попытки из очереди — no-op
        await stack.container.ingest.embed_batch(b, final_attempt=False)
    assert (await _doc(stack, user, agent, doc.id)).status == DocumentStatus.QUEUED
    await stack.drain()
    assert (await _doc(stack, user, agent, doc.id)).status == DocumentStatus.DONE


async def test_permanent_errors_fail_without_embedding(stack: Stack) -> None:
    await stack.drain()
    user, agent = await _setup(stack)
    broken = await _upload(stack, user, agent, "broken.pdf")
    scan = await _upload(stack, user, agent, "scan.pdf")
    await stack.drain()
    for d, code in ((broken, "corrupted"), (scan, "no_text_layer")):
        got = await _doc(stack, user, agent, d.id)
        assert (got.status, got.error_code) == (DocumentStatus.FAILED, code)
    assert await _points(stack, agent) == 0


async def test_content_must_match_extension(stack: Stack) -> None:
    user, agent = await _setup(stack)
    data = (FIXTURES / "tolstoy.docx").read_bytes()
    with pytest.raises(ValidationError, match="PDF"):
        await stack.container.documents.upload(user.id, agent.id, "fake.pdf", _bytes(data))


async def test_delete_document_cleans_qdrant_pg_and_file(stack: Stack) -> None:
    await stack.drain()
    user, agent = await _setup(stack)
    keep = await _upload(stack, user, agent, "chapters_cp1251.txt")
    gone = await _upload(stack, user, agent, "tolstoy.fb2")
    await stack.drain()
    kept_points = (await _doc(stack, user, agent, keep.id)).chunks_total
    path = stack.container.documents.storage.path(gone.storage_key)
    assert path.exists()

    await stack.container.documents.delete(user.id, agent.id, gone.id)
    listed = await stack.container.documents.list(user.id, agent.id)
    assert [d.id for d in listed] == [keep.id]  # deleting не показывается сразу
    await stack.drain()

    assert not path.exists()
    assert await _points(stack, agent) == kept_points
    with pytest.raises(Exception, match="document"):
        await _doc(stack, user, agent, gone.id)


async def test_delete_while_embedding_leaves_no_points(stack: Stack) -> None:
    await stack.drain()
    user, agent = await _setup(stack)
    doc = await _upload(stack, user, agent, "tolstoy.fb2")
    await stack.container.ingest.parse(stack.publisher.tasks.pop(), final_attempt=False)
    await stack.container.documents.delete(user.id, agent.id, doc.id)
    await stack.drain()  # батчи видят deleting и ничего не пишут, затем удаление
    assert await _points(stack, agent) == 0


async def test_purge_deleted_agent(stack: Stack) -> None:
    await stack.drain()
    user, agent = await _setup(stack)
    doc = await _upload(stack, user, agent, "tolstoy.docx")
    await stack.drain()
    agent_dir = stack.container.documents.storage.path(str(agent.id))
    assert agent_dir.exists()

    await stack.container.agents.delete(user.id, agent.id)
    await stack.drain()
    assert await _points(stack, agent) == 0
    assert not agent_dir.exists()
    async with stack.container.db.session() as s:
        assert await s.get(Document, doc.id) is None


async def test_sweeper_recovers_lost_messages_and_dead_workers(stack: Stack) -> None:
    await stack.drain()
    user, agent = await _setup(stack)
    lost = await _upload(stack, user, agent, "chapters_cp1251.txt")
    stuck = await _upload(stack, user, agent, "tolstoy.docx")
    stack.publisher.tasks.clear()  # брокер «потерял» сообщения
    long_ago = utcnow() - timedelta(minutes=30)
    async with stack.container.db.uow() as uow:
        await uow.session.execute(
            update(Document).where(Document.id == lost.id).values(heartbeat_at=long_ago)
        )
        # Воркер умер посреди обработки: processing с протухшим heartbeat
        await uow.session.execute(
            update(Document)
            .where(Document.id == stuck.id)
            .values(status=DocumentStatus.PROCESSING, heartbeat_at=long_ago)
        )
        await uow.commit()

    report = await stack.container.ingest.sweep()
    assert report.requeued >= 1
    assert report.restarted >= 1
    assert {t.document_id for t in stack.publisher.tasks} >= {lost.id, stuck.id}
    await stack.drain()
    for d in (lost, stuck):
        assert (await _doc(stack, user, agent, d.id)).status == DocumentStatus.DONE
    assert (await stack.container.ingest.sweep()).requeued == 0
