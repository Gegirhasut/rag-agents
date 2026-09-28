from uuid import UUID

from fastapi import APIRouter, status

from rag_agents.api.v1.schemas import ERRORS
from rag_agents.domain.auth import ApiKeyCreate, ApiKeyIssued, ApiKeyOut
from rag_agents.web.deps import ContainerDep, OwnerDep

router = APIRouter(prefix="/me/api-keys", tags=["api-keys"], responses=ERRORS)  # type: ignore[arg-type]


@router.get("")
async def list_keys(c: ContainerDep, owner: OwnerDep) -> list[ApiKeyOut]:
    return await c.auth.list_keys(owner)


@router.post("", status_code=status.HTTP_201_CREATED)
async def issue_key(data: ApiKeyCreate, c: ContainerDep, owner: OwnerDep) -> ApiKeyIssued:
    """Поле token возвращается только в этом ответе."""
    return await c.auth.issue_key(owner, data.name)


@router.delete("/{key_id}")
async def revoke_key(key_id: UUID, c: ContainerDep, owner: OwnerDep) -> ApiKeyOut:
    return await c.auth.revoke_key(owner, key_id)
