from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, Query, Response, UploadFile, status

from rag_agents.api.v1.schemas import ERRORS
from rag_agents.domain.documents import DocumentOut
from rag_agents.domain.enums import DocumentStatus
from rag_agents.services.ratelimit import Rule
from rag_agents.web.deps import ContainerDep, OwnerDep

router = APIRouter(
    prefix="/agents/{agent_id}/documents",
    tags=["documents"],
    responses=ERRORS,  # type: ignore[arg-type]
)
_READ_CHUNK = 1024 * 1024


async def _iter_upload(file: UploadFile) -> AsyncIterator[bytes]:
    while chunk := await file.read(_READ_CHUNK):
        yield chunk


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def upload_documents(
    agent_id: UUID,
    c: ContainerDep,
    owner: OwnerDep,
    files: Annotated[list[UploadFile], File()],
) -> list[DocumentOut]:
    """Файлы ставятся в очередь; статус — GET …/documents/{id}. Дубликат (тот же sha256)
    возвращает уже существующий документ."""
    rule = Rule("uploads", c.settings.rl_uploads_per_hour, 3600)
    out: list[DocumentOut] = []
    for f in files:
        try:
            await c.rate_limiter.check(rule, str(owner))
            doc, _ = await c.documents.upload(
                owner, agent_id, f.filename or "file", _iter_upload(f)
            )
            out.append(doc)
        finally:
            await f.close()
    return out


@router.get("")
async def list_documents(
    agent_id: UUID,
    c: ContainerDep,
    owner: OwnerDep,
    status_: Annotated[DocumentStatus | None, Query(alias="status")] = None,
) -> list[DocumentOut]:
    return await c.documents.list(owner, agent_id, status_)


@router.get("/{document_id}")
async def get_document(
    agent_id: UUID, document_id: UUID, c: ContainerDep, owner: OwnerDep
) -> DocumentOut:
    return await c.documents.get(owner, agent_id, document_id)


@router.post("/{document_id}/retry", status_code=status.HTTP_202_ACCEPTED)
async def retry_document(
    agent_id: UUID, document_id: UUID, c: ContainerDep, owner: OwnerDep
) -> DocumentOut:
    return await c.documents.retry(owner, agent_id, document_id)


@router.delete("/{document_id}", status_code=status.HTTP_202_ACCEPTED)
async def delete_document(
    agent_id: UUID, document_id: UUID, c: ContainerDep, owner: OwnerDep
) -> Response:
    await c.documents.delete(owner, agent_id, document_id)
    return Response(status_code=status.HTTP_202_ACCEPTED)
