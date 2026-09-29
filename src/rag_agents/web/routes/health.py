import anyio
from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
from sqlalchemy import text

from rag_agents.core import metrics
from rag_agents.web.deps import ContainerDep

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(c: ContainerDep) -> JSONResponse:
    checks: dict[str, str] = {}
    try:
        async with c.db.session() as s:
            await s.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {type(e).__name__}"
    try:
        await c.redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {type(e).__name__}"
    try:
        await c.qdrant.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"error: {type(e).__name__}"
    try:
        r = await c.ollama_http.get("/api/tags")
        checks["ollama"] = "ok" if r.is_success else f"http {r.status_code}"
    except Exception as e:
        checks["ollama"] = f"error: {type(e).__name__}"
    ok = all(v == "ok" for v in checks.values())
    return JSONResponse({"ready": ok, "checks": checks}, status_code=200 if ok else 503)


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics(c: ContainerDep) -> Response:
    """Prometheus: все процессы web и воркеров (multiprocess) + глубина очередей RabbitMQ.

    Без авторизации: локальный стенд, Prometheus ходит без сессии (ARCHITECTURE §14.3).
    """
    depths = await c.system.queue_depths()
    body, content_type = await anyio.to_thread.run_sync(
        lambda: metrics.render([metrics.QueueDepthCollector(depths)])
    )
    return Response(body, media_type=content_type)
