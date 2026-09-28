"""Клиент чтения Langfuse Public API: наблюдения и проект (для страницы «Аналитика»).

Пишем трейсы через SDK (core/observability.py), а читаем — напрямую по HTTP: SDK-клиент API
синхронный, а страница у нас async. Используются только актуальные эндпоинты:
`GET /api/public/traces` для организаций после 16.09.2026 отключён (410, ADR-9).
Metrics API сознательно не используем: на Hobby он ограничен 100 запросами в сутки,
сводку считаем по своей БД (ARCHITECTURE §14.5).
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from rag_agents.core.config import Settings


class LangfuseApiError(Exception):
    pass


class LangfuseRateLimitedError(LangfuseApiError):
    """429: лимит Public API исчерпан (у Metrics API на Hobby — 100 запросов в сутки)."""

    def __init__(self, path: str, retry_after_s: int) -> None:
        super().__init__(f"{path}: rate limited for {retry_after_s} s")
        self.retry_after_s = retry_after_s
        self.reset_at = datetime.now(UTC) + timedelta(seconds=retry_after_s)


def iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


class LangfuseReader:
    """Any в сигнатурах: ответы API — произвольный JSON, типизирует их InsightsService."""

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.enabled = settings.langfuse_active
        self.base_url = settings.langfuse_base_url.rstrip("/")
        secret = settings.langfuse_secret_key
        auth = (
            (settings.langfuse_public_key or "", secret.get_secret_value())
            if secret is not None
            else None
        )
        self._client = httpx.AsyncClient(
            base_url=f"{self.base_url}/api/public",
            auth=auth,
            timeout=httpx.Timeout(20, connect=5),
            transport=transport,
        )

    async def _get(self, path: str, params: dict[str, Any]) -> Any:
        try:
            r = await self._client.get(path, params=params)
        except httpx.HTTPError as e:
            raise LangfuseApiError(f"{path}: {e!r}") from e
        if r.status_code == httpx.codes.TOO_MANY_REQUESTS:
            raise LangfuseRateLimitedError(path, int(r.headers.get("retry-after") or 60))
        if r.is_error:
            raise LangfuseApiError(f"{path}: HTTP {r.status_code} {r.text[:200]}")
        return r.json()

    async def project_id(self) -> str:
        data = await self._get("/projects", {})
        return str(data["data"][0]["id"])

    async def observations(self, **params: Any) -> list[dict[str, Any]]:
        """GET /v2/observations: фильтры traceId, userId, sessionId, type, fromStartTime…"""
        clean = {k: v for k, v in params.items() if v is not None}
        for k, v in clean.items():
            if isinstance(v, bool):
                clean[k] = str(v).lower()
            elif isinstance(v, datetime):
                clean[k] = iso(v)
        data = await self._get("/v2/observations", clean)
        rows: list[dict[str, Any]] = data["data"]
        return rows

    async def aclose(self) -> None:
        await self._client.aclose()
