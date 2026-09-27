from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

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
