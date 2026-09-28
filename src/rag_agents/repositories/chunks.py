import hashlib
from uuid import UUID

from sqlalchemy import delete, func, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agents.domain.documents import ChunkDraft, ChunkOut, TocItem
from rag_agents.models.entities import Chunk

_INSERT_BATCH = 500


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def delete_for_document(self, agent_id: UUID, document_id: UUID) -> None:
        """Перед повторным парсингом: следующая вставка не плодит дубли."""
        await self.s.execute(
            delete(Chunk).where(Chunk.document_id == document_id, Chunk.agent_id == agent_id)
        )

    async def insert(
        self,
        agent_id: UUID,
        document_id: UUID,
        chunks: list[tuple[UUID, ChunkDraft]],
    ) -> None:
        rows = [
            {
                "id": cid,
                "agent_id": agent_id,
                "document_id": document_id,
                "ord": d.ord,
                "section_path": d.section_path,
                "chapter_title": d.chapter_title,
                "text": d.text,
                "embed_text": d.embed_text,
                "token_count": d.token_count,
                "char_start": d.char_start,
                "char_end": d.char_end,
                "page_from": d.page_from,
                "page_to": d.page_to,
                "content_hash": hashlib.sha1(d.text.encode(), usedforsecurity=False).digest(),
            }
            for cid, d in chunks
        ]
        for i in range(0, len(rows), _INSERT_BATCH):
            await self.s.execute(insert(Chunk), rows[i : i + _INSERT_BATCH])

    async def range(
        self, agent_id: UUID, document_id: UUID, ord_from: int, ord_to: int
    ) -> list[ChunkOut]:
        rows = await self.s.scalars(
            select(Chunk)
            .where(
                Chunk.agent_id == agent_id,
                Chunk.document_id == document_id,
                Chunk.ord.between(ord_from, ord_to),
            )
            .order_by(Chunk.ord)
        )
        return [ChunkOut.model_validate(c) for c in rows]

    async def toc(self, agent_id: UUID, document_id: UUID) -> list[TocItem]:
        """Оглавление по фактическим чанкам: секции в порядке появления."""
        stmt = (
            select(
                Chunk.section_path,
                func.min(Chunk.ord),
                func.count(),
                func.sum(Chunk.token_count),
            )
            .where(Chunk.agent_id == agent_id, Chunk.document_id == document_id)
            .group_by(Chunk.section_path)
            .order_by(func.min(Chunk.ord))
        )
        return [
            TocItem(section_path=list(p), first_ord=o, chunks=n, tokens=int(t or 0))
            for p, o, n, t in (await self.s.execute(stmt)).all()
        ]

    async def count(self, agent_id: UUID, document_id: UUID) -> int:
        n = await self.s.scalar(
            select(func.count()).where(Chunk.agent_id == agent_id, Chunk.document_id == document_id)
        )
        return int(n or 0)
