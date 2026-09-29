"""Страница «Качество» (/eval): прогоны eval агента, графики метрик и сравнение с CI.

Прогоны запускаются из CLI (`make eval`), здесь только чтение. Прогон доступен через своего
агента: чужой агент или прогон другого агента → 404 (tests/integration/test_isolation.py).
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from rag_agents.domain.agents import AgentListItem
from rag_agents.services.errors import NotFoundError
from rag_agents.web.deps import ContainerDep, OwnerDep
from rag_agents.web.eval_view import (
    CATEGORY_LABELS,
    METRICS,
    build_page,
    default_base,
    run_headline,
)
from rag_agents.web.templating import templates

router = APIRouter(prefix="/eval")


async def _render(
    request: Request,
    c: ContainerDep,
    owner: UUID,
    agent_id: UUID,
    run_id: UUID | None,
    base_id: UUID | None,
) -> Response:
    not_found = HTTPException(404, "Не найдено")
    try:
        runs = await c.evals.owned_runs(owner, agent_id)
    except NotFoundError as e:
        raise not_found from e
    agents = await c.agents.list(owner)
    agent = next((a for a in agents if a.agent.id == agent_id), None)
    if agent is None:
        raise not_found
    finished = [r for r in runs if r.finished_at is not None]
    if run_id is None and not finished:
        return _page(request, agents, agent, runs=runs, page=None)
    # Базу сравнения тоже читаем через агента: чужой ?base= → 404, как и чужой прогон
    try:
        run, items = await c.evals.owned_run(owner, agent_id, run_id or finished[0].id)
        base_ref = default_base(run, runs) if base_id is None else None
        base_run_id = base_id or (base_ref.id if base_ref else None)
        base = None
        if base_run_id is not None and base_run_id != run.id:
            base = await c.evals.owned_run(owner, agent_id, base_run_id)
    except NotFoundError as e:
        raise not_found from e
    page = await build_page(run, items, runs, base)
    return _page(request, agents, agent, runs=runs, page=page)


def _page(
    request: Request,
    agents: list[AgentListItem],
    agent: AgentListItem,
    **ctx: object,
) -> Response:
    return templates.TemplateResponse(
        request,
        "pages/eval.html",
        {
            "agents": agents,
            "agent": agent,
            "labels": METRICS,
            "category_labels": CATEGORY_LABELS,
            "headline": run_headline,
            **ctx,
        },
    )


@router.get("", response_class=HTMLResponse)
async def eval_index(request: Request, c: ContainerDep, owner: OwnerDep) -> Response:
    """Первый агент, у которого есть прогоны (или просто первый — с подсказкой, как запустить)."""
    agents = await c.agents.list(owner)
    if not agents:
        return templates.TemplateResponse(request, "pages/eval.html", {"agents": [], "agent": None})
    target = agents[0]
    for a in agents:
        if await c.evals.owned_runs(owner, a.agent.id, limit=1):
            target = a
            break
    return RedirectResponse(f"/eval/agents/{target.agent.id}", status_code=303)


@router.get("/agents/{agent_id}", response_class=HTMLResponse)
async def eval_agent(
    request: Request,
    agent_id: UUID,
    c: ContainerDep,
    owner: OwnerDep,
) -> Response:
    return await _render(request, c, owner, agent_id, None, None)


@router.get("/agents/{agent_id}/runs/{run_id}", response_class=HTMLResponse)
async def eval_run(
    request: Request,
    agent_id: UUID,
    run_id: UUID,
    c: ContainerDep,
    owner: OwnerDep,
    base: Annotated[UUID | None, Query()] = None,
) -> Response:
    """?base=<run_id> — с каким прогоном сравнивать (по умолчанию предыдущий на том же датасете)."""
    return await _render(request, c, owner, agent_id, run_id, base)
