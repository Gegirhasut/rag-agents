"""Выгрузка eval в Langfuse Datasets: поминутный rate limit пережидается, суточный — нет."""

import httpx
import pytest

from rag_agents.core import langfuse_api
from rag_agents.core.config import Settings
from rag_agents.core.langfuse_api import LangfuseDatasets, LangfuseRateLimitedError


def client(responses: list[httpx.Response]) -> tuple[LangfuseDatasets, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses.pop(0)

    settings = Settings(_env_file=None, app_env="dev")
    return LangfuseDatasets(settings, transport=httpx.MockTransport(handler)), seen


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waits: list[float] = []

    async def fake_sleep(s: float) -> None:
        waits.append(s)

    monkeypatch.setattr(langfuse_api.asyncio, "sleep", fake_sleep)
    return waits


async def test_short_rate_limit_is_waited_out(slept: list[float]) -> None:
    """Регрессия 2026-09-29: ~120 POST подряд → 429 retry-after 52 с → экспорт терялся."""
    lf, seen = client(
        [httpx.Response(429, headers={"retry-after": "52"}), httpx.Response(200, json={})]
    )
    await lf.upsert_dataset("tolstoy", "d")
    assert slept == [52]
    assert len(seen) == 2


async def test_long_rate_limit_is_raised(slept: list[float]) -> None:
    lf, _ = client([httpx.Response(429, headers={"retry-after": "86400"})])
    with pytest.raises(LangfuseRateLimitedError):
        await lf.upsert_dataset("tolstoy", "d")
    assert slept == []


async def test_gives_up_after_three_waits(slept: list[float]) -> None:
    lf, seen = client([httpx.Response(429, headers={"retry-after": "5"}) for _ in range(4)])
    with pytest.raises(LangfuseRateLimitedError):
        await lf.upsert_dataset("tolstoy", "d")
    assert slept == [5, 5, 5]
    assert len(seen) == 4
