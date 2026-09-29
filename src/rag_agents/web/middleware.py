import re
import time
import uuid

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag_agents.core import metrics

log = structlog.get_logger()

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
# Пробы и статика шумят в логах и ничего не говорят о поведении приложения
_QUIET_PREFIXES = ("/healthz", "/readyz", "/static/", "/metrics")


def route_label(scope: Scope) -> str:
    """Шаблон роута (/agents/{agent_id}), а не путь: иначе каждый id — новый временной ряд.

    FastAPI 0.141 кладёт в scope роут вложенного роутера без префиксов внешних
    (`/agents/{agent_id}/query` вместо `/api/v1/agents/...`). Префикс восстанавливаем по пути:
    ищем, с какого `/` хвост пути совпадает с regex роута. Префиксы у нас статические.
    """
    route = scope.get("route")
    template = getattr(route, "path", None)
    regex = getattr(route, "path_regex", None)
    path: str = scope["path"]
    if isinstance(template, str):
        if regex is not None:
            for i, ch in enumerate(path):
                if ch == "/" and regex.match(path[i:]):
                    return path[:i] + template
        return template
    if path.startswith("/static/"):
        return "/static"
    return "unmatched"


class RequestContextMiddleware:
    """request_id в contextvars structlog + заголовок ответа + одна строка лога на запрос.

    Чистый ASGI, а не BaseHTTPMiddleware: тот буферизует контекст и мешает SSE-стримам.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = dict(scope["headers"]).get(REQUEST_ID_HEADER.lower().encode(), b"").decode()
        request_id = incoming if _VALID_ID.match(incoming) else uuid.uuid4().hex
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        status = 500
        t0 = time.monotonic()

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = [
                    *message.get("headers", []),
                    (REQUEST_ID_HEADER.lower().encode(), request_id.encode()),
                ]
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            path: str = scope["path"]
            elapsed = time.monotonic() - t0
            route, method = route_label(scope), scope["method"]
            metrics.HTTP_REQUESTS.labels(route, method, str(status)).inc()
            metrics.HTTP_SECONDS.labels(route, method).observe(elapsed)
            if not path.startswith(_QUIET_PREFIXES):
                log.info(
                    "http.request",
                    method=method,
                    path=path,
                    status=status,
                    duration_ms=int(elapsed * 1000),
                )
            structlog.contextvars.clear_contextvars()
