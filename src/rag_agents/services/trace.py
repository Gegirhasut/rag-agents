"""Шина живых событий для страницы «Под капотом» (Redis pub/sub + короткая история).

Web и воркеры публикуют сюда шаги пайплайна, страница /system подписывается через SSE.
Pub/sub, а не очередь: событие без подписчика просто теряется, и это нормально —
это визуализация, а не аудит. Ошибка шины никогда не ломает основной сценарий.
"""

import time
from collections.abc import AsyncIterator
from typing import Any

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from rag_agents.domain.system import Node, TraceEvent

log = structlog.get_logger()

CHANNEL = "trace:events"
HISTORY_KEY = "trace:history"
HISTORY_SIZE = 150
_HISTORY_TTL_S = 24 * 3600


class TraceBus:
    def __init__(self, redis: Redis, *, enabled: bool = True) -> None:
        self.redis = redis
        self.enabled = enabled

    async def emit(
        self,
        kind: str,
        src: Node,
        dst: Node | None,
        label: str,
        *,
        agent_id: object | None = None,
        **data: Any,  # Any: произвольные JSON-сериализуемые детали события
    ) -> None:
        """Публикует событие; при недоступном Redis молча (с warning) пропускает."""
        if not self.enabled:
            return
        ev = TraceEvent(
            ts=time.time(),
            kind=kind,
            src=src,
            dst=dst,
            label=label,
            agent_id=str(agent_id) if agent_id is not None else None,
            data=data,
        )
        raw = ev.model_dump_json()
        try:
            async with self.redis.pipeline(transaction=False) as pipe:
                pipe.publish(CHANNEL, raw)
                pipe.lpush(HISTORY_KEY, raw)
                pipe.ltrim(HISTORY_KEY, 0, HISTORY_SIZE - 1)
                pipe.expire(HISTORY_KEY, _HISTORY_TTL_S)
                await pipe.execute()
        except RedisError as e:
            log.warning("trace.emit_failed", kind=kind, error=str(e))

    async def history(self, limit: int = 60) -> list[TraceEvent]:
        """Последние события, от старых к новым."""
        raw = await self.redis.lrange(HISTORY_KEY, 0, limit - 1)
        return [TraceEvent.model_validate_json(r) for r in reversed(raw)]

    async def subscribe(self) -> AsyncIterator[TraceEvent]:
        """Бесконечный поток событий; закрывается вместе с SSE-соединением."""
        async with self.redis.pubsub() as pubsub:
            await pubsub.subscribe(CHANNEL)
            async for msg in pubsub.listen():
                if msg["type"] == "message":
                    yield TraceEvent.model_validate_json(msg["data"])
