"""Страница «Аналитика»: сводка из нашей БД, трейсы — из Langfuse (ARCHITECTURE §14.5)."""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response

from rag_agents.domain.insights import Period
from rag_agents.services.errors import NotFoundError
from rag_agents.services.insights import InsightsDisabledError, InsightsUnavailableError
from rag_agents.web.deps import ContainerDep, OwnerDep
from rag_agents.web.templating import templates

router = APIRouter(prefix="/insights")
# Свежий трейс появляется в API Langfuse через секунды; ждём до ~30 с, потом «не найден»
MAX_TRACE_POLLS = 10
PeriodQuery = Annotated[Period, Query()]


def _unavailable(request: Request, error: str | None = None) -> Response:
    return templates.TemplateResponse(
        request, "fragments/insights_unavailable.html", {"error": error}
    )


@router.get("", response_class=HTMLResponse)
async def insights_page(
    request: Request, c: ContainerDep, period: PeriodQuery = Period.DAY
) -> Response:
    return templates.TemplateResponse(
        request,
        "pages/insights.html",
        {
            "period": period,
            "periods": list(Period),
            "enabled": c.insights.enabled,
            "project_url": await c.insights.project_url(),
        },
    )


@router.get("/overview", response_class=HTMLResponse)
async def insights_overview(
    request: Request, c: ContainerDep, owner: OwnerDep, period: PeriodQuery = Period.DAY
) -> Response:
    """Сводка из PG: работает без Langfuse и без лимитов его API."""
    data = await c.insights.overview(owner, period)
    return templates.TemplateResponse(request, "fragments/insights_overview.html", {"o": data})


@router.get("/traces/{trace_id}", response_class=HTMLResponse)
async def trace_page(
    request: Request,
    trace_id: str,
    c: ContainerDep,
    owner: OwnerDep,
    attempt: Annotated[int, Query(ge=0, le=MAX_TRACE_POLLS)] = 0,
) -> Response:
    """Полная страница трейса; для HTMX — только фрагмент (с автоповтором, пока Langfuse
    обрабатывает свежий трейс)."""
    fragment = bool(request.headers.get("HX-Request"))
    try:
        detail = await c.insights.trace(owner, trace_id)
    except NotFoundError as e:
        raise HTTPException(404, "Трейс не найден") from e
    except InsightsDisabledError:
        return _unavailable(request)
    except InsightsUnavailableError as e:
        return _unavailable(request, str(e))
    ctx = {
        "t": detail,
        "trace_id": trace_id,
        "attempt": attempt,
        "give_up": attempt >= MAX_TRACE_POLLS,
    }
    if fragment:
        return templates.TemplateResponse(request, "fragments/insights_trace.html", ctx)
    return templates.TemplateResponse(request, "pages/insights_trace.html", ctx)


@router.get("/sessions/{session_id}", response_class=HTMLResponse)
async def session_page(
    request: Request, session_id: str, c: ContainerDep, owner: OwnerDep
) -> Response:
    """Все трейсы сессии: чат (session = chat_id) или документ (document-{id})."""
    error: str | None = None
    try:
        rows = await c.insights.session(owner, session_id)
    except InsightsDisabledError:
        rows, error = [], "Langfuse не настроен"
    except InsightsUnavailableError as e:
        rows, error = [], str(e)
    return templates.TemplateResponse(
        request,
        "pages/insights_session.html",
        {
            "session_id": session_id,
            "rows": rows,
            "error": error,
            "project_url": await c.insights.project_url(),
        },
    )
