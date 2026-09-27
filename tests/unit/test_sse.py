import asyncio
from collections.abc import AsyncIterator

import pytest

from rag_agents.web.sse import format_sse, with_heartbeat


def test_multiline_data_becomes_multiple_data_lines() -> None:
    assert format_sse("token", "a\nb", event_id=3) == "id: 3\nevent: token\ndata: a\ndata: b\n\n"


async def test_heartbeat_emitted_during_pause_and_events_preserved() -> None:
    async def slow() -> AsyncIterator[str]:
        yield "first"
        await asyncio.sleep(0.25)
        yield "second"

    out = [item async for item in with_heartbeat(slow(), interval=0.1)]
    assert out[0] == "first"
    assert out[-1] == "second"
    assert ": ping\n\n" in out


async def test_heartbeat_propagates_source_error() -> None:
    async def broken() -> AsyncIterator[str]:
        yield "x"
        raise RuntimeError("boom")

    got: list[str] = []

    async def consume() -> None:
        async for item in with_heartbeat(broken(), interval=1):
            got.append(item)

    with pytest.raises(RuntimeError, match="boom"):
        await consume()
    assert got == ["x"]
