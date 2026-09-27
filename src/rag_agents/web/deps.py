from typing import Annotated
from uuid import UUID

from fastapi import Depends, Request

from rag_agents.container import Container


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


def get_owner_id(request: Request) -> UUID:
    owner: UUID = request.app.state.owner_id
    return owner


ContainerDep = Annotated[Container, Depends(get_container)]
OwnerDep = Annotated[UUID, Depends(get_owner_id)]
