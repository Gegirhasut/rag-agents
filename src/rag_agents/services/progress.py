from uuid import UUID

from redis.asyncio import Redis

from rag_agents.domain.enums import IngestStage

_TTL_S = 3600

# Диапазоны прогресса по этапам (ARCHITECTURE §6.4)
STAGE_RANGE: dict[IngestStage, tuple[int, int]] = {
    IngestStage.PARSING: (0, 10),
    IngestStage.CHUNKING: (10, 20),
    IngestStage.EMBEDDING: (20, 98),
    IngestStage.FINALIZING: (98, 100),
}


def stage_progress(stage: IngestStage, fraction: float) -> int:
    lo, hi = STAGE_RANGE[stage]
    return int(lo + (hi - lo) * max(0.0, min(1.0, fraction)))


class ProgressStore:
    """Hot-прогресс ingest в Redis: часто пишет воркер, читает polling-эндпоинт."""

    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    @staticmethod
    def _key(document_id: UUID) -> str:
        return f"ingest:{document_id}"

    async def set(self, document_id: UUID, stage: IngestStage, fraction: float = 0.0) -> None:
        key = self._key(document_id)
        async with self.redis.pipeline(transaction=False) as pipe:
            pipe.hset(
                key, mapping={"stage": stage.value, "progress": stage_progress(stage, fraction)}
            )
            pipe.expire(key, _TTL_S)
            await pipe.execute()

    async def get_many(self, document_ids: list[UUID]) -> dict[UUID, int]:
        if not document_ids:
            return {}
        async with self.redis.pipeline(transaction=False) as pipe:
            for did in document_ids:
                pipe.hget(self._key(did), "progress")
            values = await pipe.execute()
        return {did: int(v) for did, v in zip(document_ids, values, strict=True) if v is not None}

    async def clear(self, document_id: UUID) -> None:
        await self.redis.delete(self._key(document_id))
