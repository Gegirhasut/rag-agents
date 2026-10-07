"""Страница «Под капотом»: живая схема сервисов, карта векторов, состояние инфраструктуры."""

from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from rag_agents.container import Container
from rag_agents.domain.eval import EvalRunOut
from rag_agents.domain.system import SystemSnapshot
from rag_agents.services.errors import NotFoundError
from rag_agents.web.deps import ContainerDep, OwnerDep, require_admin
from rag_agents.web.sse import format_sse, with_heartbeat
from rag_agents.web.system_catalog import EDGES, LIBS, NODE_H, NODE_W, NODES
from rag_agents.web.templating import templates

# Только администраторы: живые события пайплайна (/system/events) общие для всех пользователей
router = APIRouter(prefix="/system", dependencies=[Depends(require_admin)])


def _badges(
    s: SystemSnapshot, last_eval: EvalRunOut | None = None, *, tracing: bool = False
) -> dict[str, str]:
    """Короткие живые подписи под узлами схемы."""
    points = sum(c.points for c in s.qdrant.collections)
    ready = sum(q.ready for q in s.celery.queues if not q.is_dlq)
    unacked = sum(q.unacked for q in s.celery.queues)
    dlq = sum(q.ready for q in s.celery.queues if q.is_dlq)
    active = sum(len(t) for t in (s.celery.workers or {}).values())
    loaded = [m.name for m in s.ollama.models if m.loaded]
    if not s.celery.ok or s.celery.workers is None:
        worker = "нет данных"
    elif not s.celery.workers:
        worker = "воркеров нет!"
    else:
        worker = f"занят: {active} задач" if active else "простаивает"
    return {
        "browser": "вы здесь",
        "web": "2 процесса uvicorn",
        "postgres": s.postgres.db_size or "недоступен",
        "redis": s.redis.used_memory or "недоступен",
        "rabbitmq": (
            f"ждут {ready} · в работе {unacked}" + (f" · DLQ {dlq}" if dlq else "")
            if s.celery.ok
            else "недоступен"
        ),
        "uploads": "docker volume",
        "worker": worker,
        "qdrant": f"{points} точек" if s.qdrant.ok else "недоступен",
        "ollama": (
            ("в памяти: " + ", ".join(loaded) if loaded else "модель выгружена")
            if s.ollama.ok
            else "недоступна"
        ),
        "llm": s.llm.model + ("" if s.llm.key_configured else " · нет ключа"),
        "eval": _eval_badge(last_eval),
        "langfuse": "трейсы уходят фоном" if tracing else "выключен: нет ключей",
    }


def _eval_badge(run: EvalRunOut | None) -> str:
    if run is None:
        return "прогонов нет · make eval"
    hit = ((run.metrics or {}).get("metrics", {}).get("hit@8") or {}).get("mean")
    when = run.started_at.strftime("%d.%m")
    return f"{run.config_name} {when}" + (f" · hit@8 {hit:.2f}" if hit is not None else "")


async def _last_eval(c: Container, owner: UUID) -> EvalRunOut | None:
    """Последний завершённый прогон по агентам владельца (для подписи узла eval)."""
    runs = []
    for a in await c.agents.list(owner):
        runs += [r for r in await c.evals.owned_runs(owner, a.agent.id, limit=5) if r.finished_at]
    return max(runs, key=lambda r: r.started_at, default=None)


@router.get("", response_class=HTMLResponse)
async def system_page(
    request: Request, c: ContainerDep, owner: OwnerDep, agent: UUID | None = None
) -> Response:
    """?agent=<id> — какой агент выбран для карты векторов и вопроса."""
    agents = await c.agents.list(owner)
    selected = next((a for a in agents if a.agent.id == agent), agents[0] if agents else None)
    history = await c.trace.history()
    return templates.TemplateResponse(
        request,
        "pages/system.html",
        {
            "agents": agents,
            "selected": selected,
            "nodes": NODES,
            "edges": EDGES,
            "libs": LIBS,
            "node_w": NODE_W,
            "node_h": NODE_H,
            "history": [e.model_dump(mode="json", exclude={"data"}) for e in history],
            "trace_enabled": c.settings.trace_enabled,
        },
    )


@router.get("/stats", response_class=HTMLResponse)
async def system_stats(request: Request, c: ContainerDep, owner: OwnerDep) -> Response:
    snap = await c.system.snapshot(owner)
    last_eval = await _last_eval(c, owner)
    return templates.TemplateResponse(
        request,
        "fragments/system_stats.html",
        {
            "s": snap,
            "badges": _badges(snap, last_eval, tracing=c.tracer.enabled),
            "host": request.url.hostname,
            "admin_by_key": {u.key: u for u in snap.admin_uis},
        },
    )


@router.get("/events")
async def system_events(c: ContainerDep) -> Response:
    async def sse() -> AsyncIterator[str]:
        async for ev in c.trace.subscribe():
            yield format_sse("trace", ev.model_dump_json())

    return StreamingResponse(
        with_heartbeat(sse()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/agents/{agent_id}/vector-map")
async def vector_map(agent_id: UUID, c: ContainerDep, owner: OwnerDep) -> Response:
    try:
        vmap = await c.system.vector_map(owner, agent_id)
    except NotFoundError as e:
        raise HTTPException(404, "Агент не найден") from e
    if vmap is None:  # строит воркер; браузер повторит запрос
        return JSONResponse({"status": "building"}, status_code=202)
    return JSONResponse(vmap.model_dump(mode="json"))


@router.get("/agents/{agent_id}/points/{point_id}")
async def vector_point(
    agent_id: UUID, point_id: UUID, c: ContainerDep, owner: OwnerDep
) -> Response:
    try:
        detail = await c.system.point(owner, agent_id, point_id)
    except NotFoundError as e:
        raise HTTPException(404, "Точка не найдена") from e
    return JSONResponse(detail.model_dump(mode="json"))
