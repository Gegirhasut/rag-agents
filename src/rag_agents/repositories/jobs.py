from uuid import UUID

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agents.core.ids import utcnow
from rag_agents.domain.enums import JobStatus
from rag_agents.models.entities import EmbedBatch, IngestJob


class JobRepository:
    """ingest_jobs и embed_batches: идемпотентный учёт фан-аута (ARCHITECTURE §10.4)."""

    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def start(self, job_id: UUID) -> None:
        await self.s.execute(
            update(IngestJob)
            .where(IngestJob.id == job_id)
            .values(status=JobStatus.RUNNING, attempts=IngestJob.attempts + 1)
        )

    async def create_batches(self, job_id: UUID, ranges: list[tuple[int, int]]) -> None:
        """ranges — (ord_from, ord_to) включительно; batch_no с 1."""
        await self.s.execute(delete(EmbedBatch).where(EmbedBatch.job_id == job_id))
        if ranges:
            await self.s.execute(
                insert(EmbedBatch),
                [
                    {"job_id": job_id, "batch_no": n, "ord_from": a, "ord_to": b}
                    for n, (a, b) in enumerate(ranges, start=1)
                ],
            )

    async def get_batch(self, job_id: UUID, batch_no: int) -> tuple[int, int, JobStatus] | None:
        row = (
            await self.s.execute(
                select(EmbedBatch.ord_from, EmbedBatch.ord_to, EmbedBatch.status).where(
                    EmbedBatch.job_id == job_id, EmbedBatch.batch_no == batch_no
                )
            )
        ).first()
        return (row[0], row[1], row[2]) if row else None

    async def complete_batch(
        self, job_id: UUID, batch_no: int, *, embed_ms: int, upsert_ms: int
    ) -> bool:
        """done ставит только первый исполнитель: повторная доставка не даст второго +1."""
        found = await self.s.scalar(
            update(EmbedBatch)
            .where(
                EmbedBatch.job_id == job_id,
                EmbedBatch.batch_no == batch_no,
                EmbedBatch.status != JobStatus.DONE,
            )
            .values(
                status=JobStatus.DONE,
                attempts=EmbedBatch.attempts + 1,
                embed_ms=embed_ms,
                upsert_ms=upsert_ms,
            )
            .returning(EmbedBatch.batch_no)
        )
        return found is not None

    async def fail_batch(self, job_id: UUID, batch_no: int) -> None:
        await self.s.execute(
            update(EmbedBatch)
            .where(
                EmbedBatch.job_id == job_id,
                EmbedBatch.batch_no == batch_no,
                EmbedBatch.status != JobStatus.DONE,
            )
            .values(status=JobStatus.FAILED, attempts=EmbedBatch.attempts + 1)
        )

    async def batch_timings(self, job_id: UUID) -> tuple[int, int]:
        row = (
            await self.s.execute(
                select(
                    func.coalesce(func.sum(EmbedBatch.embed_ms), 0),
                    func.coalesce(func.sum(EmbedBatch.upsert_ms), 0),
                ).where(EmbedBatch.job_id == job_id)
            )
        ).one()
        return int(row[0]), int(row[1])

    async def set_stats(self, job_id: UUID, stats: dict[str, int]) -> None:
        await self.s.execute(update(IngestJob).where(IngestJob.id == job_id).values(stats=stats))

    async def get_stats(self, job_id: UUID) -> dict[str, int]:
        stats = await self.s.scalar(select(IngestJob.stats).where(IngestJob.id == job_id))
        return dict(stats or {})

    async def finish(self, job_id: UUID, status: JobStatus, error: str | None = None) -> None:
        await self.s.execute(
            update(IngestJob)
            .where(IngestJob.id == job_id)
            .values(status=status, error=error, finished_at=utcnow())
        )
