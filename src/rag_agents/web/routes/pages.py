from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import ValidationError as PydanticValidationError

from rag_agents.domain.agents import AgentCreate
from rag_agents.services.errors import NotFoundError
from rag_agents.web.deps import ContainerDep, OwnerDep, PrincipalDep
from rag_agents.web.routes.documents import has_inflight
from rag_agents.web.templating import templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def agents_list(request: Request, c: ContainerDep, owner: OwnerDep) -> Response:
    items = await c.agents.list(owner)
    return templates.TemplateResponse(request, "pages/agents_list.html", {"items": items})


@router.get("/agents/new", response_class=HTMLResponse)
async def agent_new(request: Request, _: PrincipalDep) -> Response:
    return templates.TemplateResponse(request, "pages/agent_new.html", {"errors": [], "form": {}})


@router.post("/agents")
async def agent_create(
    request: Request,
    c: ContainerDep,
    owner: OwnerDep,
    name: Annotated[str, Form()] = "",
    description: Annotated[str, Form()] = "",
    persona_prompt: Annotated[str, Form()] = "",
) -> Response:
    form = {"name": name, "description": description, "persona_prompt": persona_prompt}
    try:
        data = AgentCreate(
            name=name, description=description, persona_prompt=persona_prompt or None
        )
    except PydanticValidationError as e:
        errors = [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]
        return templates.TemplateResponse(
            request, "pages/agent_new.html", {"errors": errors, "form": form}, status_code=422
        )
    agent = await c.agents.create(owner, data)
    return RedirectResponse(f"/agents/{agent.id}", status_code=303)


@router.get("/agents/{agent_id}", response_class=HTMLResponse)
async def agent_detail(
    request: Request, agent_id: UUID, c: ContainerDep, owner: OwnerDep
) -> Response:
    try:
        agent = await c.agents.get(owner, agent_id)
        docs = await c.documents.list(owner, agent_id)
        messages = await c.query.history(owner, agent_id)
    except NotFoundError as e:
        raise HTTPException(404, "Агент не найден") from e
    return templates.TemplateResponse(
        request,
        "pages/agent_detail.html",
        {
            "agent": agent,
            "docs": docs,
            "polling": has_inflight(docs),
            "messages": messages,
        },
    )
