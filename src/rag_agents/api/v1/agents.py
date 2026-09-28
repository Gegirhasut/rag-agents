from uuid import UUID

from fastapi import APIRouter, Response, status

from rag_agents.api.v1.schemas import ERRORS, AgentItem, Page
from rag_agents.domain.agents import AgentCreate, AgentOut, AgentUpdate
from rag_agents.web.deps import ContainerDep, OwnerDep

router = APIRouter(prefix="/agents", tags=["agents"], responses=ERRORS)  # type: ignore[arg-type]


@router.get("")
async def list_agents(c: ContainerDep, owner: OwnerDep) -> Page[AgentItem]:
    items = [
        AgentItem(
            **i.agent.model_dump(),
            documents_total=i.documents_total,
            documents_done=i.documents_done,
        )
        for i in await c.agents.list(owner)
    ]
    return Page(items=items, total=len(items))


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_agent(data: AgentCreate, c: ContainerDep, owner: OwnerDep) -> AgentOut:
    return await c.agents.create(owner, data)


@router.get("/{agent_id}")
async def get_agent(agent_id: UUID, c: ContainerDep, owner: OwnerDep) -> AgentOut:
    return await c.agents.get(owner, agent_id)


@router.patch("/{agent_id}")
async def update_agent(
    agent_id: UUID, data: AgentUpdate, c: ContainerDep, owner: OwnerDep
) -> AgentOut:
    return await c.agents.update(owner, agent_id, data)


@router.delete("/{agent_id}", status_code=status.HTTP_202_ACCEPTED)
async def delete_agent(agent_id: UUID, c: ContainerDep, owner: OwnerDep) -> Response:
    """Агент сразу пропадает из API и UI; данные в Qdrant и файлы чистит фоновая задача."""
    await c.agents.delete(owner, agent_id)
    return Response(status_code=status.HTTP_202_ACCEPTED)
