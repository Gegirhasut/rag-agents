import asyncio
import contextlib
from collections.abc import AsyncIterator

HEARTBEAT_S = 15.0


def format_sse(event: str, data: str, event_id: int | None = None) -> str:
    """Одно SSE-событие. Многострочные данные — отдельными строками data: (спецификация SSE)."""
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.extend(f"data: {line}" for line in data.split("\n"))
    return "\n".join(lines) + "\n\n"


async def with_heartbeat(
    source: AsyncIterator[str], interval: float = HEARTBEAT_S
) -> AsyncIterator[str]:
    """Пробрасывает события источника, а в паузах длиннее interval шлёт комментарий-пинг.

    Пинги не дают прокси и браузеру оборвать соединение, пока идёт retrieval.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def pump() -> None:
        try:
            async for item in source:
                await queue.put(item)
        finally:
            await queue.put(None)

    task = asyncio.create_task(pump())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=interval)
            except TimeoutError:
                yield ": ping\n\n"
                continue
            if item is None:
                break
            yield item
        await task  # пробросить исключение источника, если было
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
