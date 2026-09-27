import hashlib
from uuid import UUID

from sqlalchemy import delete, insert
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agents.domain.documents import ChunkDraft
from rag_agents.models.entities import Chunk

_INSERT_BATCH = 500


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def replace_for_document(
        self,
        agent_id: UUID,
        document_id: UUID,
        chunks: list[tuple[UUID, ChunkDraft]],
    ) -> None:
        """DELETE + INSERT в одной транзакции: повторный парсинг не плодит дубли."""
        await self.s.execute(
            delete(Chunk).where(Chunk.document_id == document_id, Chunk.agent_id == agent_id)
        )
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
                "content_hash": hashlib.sha1(d.text.encode(), usedforsecurity=False).digest(),
            }
            for cid, d in chunks
        ]
        for i in range(0, len(rows), _INSERT_BATCH):
            await self.s.execute(insert(Chunk), rows[i : i + _INSERT_BATCH])
