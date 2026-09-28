from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response

from rag_agents.domain.documents import DocumentOut
from rag_agents.services.errors import NotFoundError, ValidationError
from rag_agents.services.ratelimit import RateLimitedError, Rule
from rag_agents.web.deps import ContainerDep, OwnerDep
from rag_agents.web.templating import templates

router = APIRouter()
_READ_CHUNK = 1024 * 1024
# HTMX: ответ 286 останавливает polling (hx-trigger="every Ns")
HTMX_STOP_POLLING = 286


def has_inflight(docs: list[DocumentOut]) -> bool:
    return any(not d.status.is_terminal for d in docs)


async def _iter_upload(file: UploadFile) -> AsyncIterator[bytes]:
    while chunk := await file.read(_READ_CHUNK):
        yield chunk


@router.post("/agents/{agent_id}/documents", response_class=HTMLResponse)
async def upload_documents(
    request: Request,
    agent_id: UUID,
    c: ContainerDep,
    owner: OwnerDep,
    files: Annotated[list[UploadFile], File()],
) -> Response:
    errors: list[str] = []
    notes: list[str] = []
    rule = Rule("uploads", c.settings.rl_uploads_per_hour, 3600)
    for f in files:
        name = f.filename or "file"
        try:
            await c.rate_limiter.check(rule, str(owner))
            _, created = await c.documents.upload(owner, agent_id, name, _iter_upload(f))
            if not created:
                notes.append(f"«{name}» уже загружен в этого агента — пропущен")
        except ValidationError as e:
            errors.append(f"«{name}»: {e}")
        except RateLimitedError as e:
            errors.append(f"«{name}»: лимит загрузок, повторите через {e.retry_after_s} с")
        except NotFoundError as e:
            raise HTTPException(404, "Агент не найден") from e
        finally:
            await f.close()
    docs = await c.documents.list(owner, agent_id)
    return templates.TemplateResponse(
        request,
        "fragments/documents_panel.html",
        {
            "agent_id": agent_id,
            "docs": docs,
            "polling": has_inflight(docs),
            "errors": errors,
            "notes": notes,
        },
    )


@router.get("/agents/{agent_id}/documents/status", response_class=HTMLResponse)
async def documents_status(
    request: Request, agent_id: UUID, c: ContainerDep, owner: OwnerDep
) -> Response:
    try:
        docs = await c.documents.list(owner, agent_id)
    except NotFoundError as e:
        raise HTTPException(404, "Агент не найден") from e
    polling = has_inflight(docs)
    return templates.TemplateResponse(
        request,
        "fragments/documents_table.html",
        {"agent_id": agent_id, "docs": docs, "polling": polling},
        status_code=200 if polling else HTMX_STOP_POLLING,
    )


@router.post("/agents/{agent_id}/documents/{document_id}/retry", response_class=HTMLResponse)
async def retry_document(
    request: Request, agent_id: UUID, document_id: UUID, c: ContainerDep, owner: OwnerDep
) -> Response:
    """Повтор из failed; в ответ — таблица целиком: она снова включит polling."""
    await c.documents.retry(owner, agent_id, document_id)
    docs = await c.documents.list(owner, agent_id)
    return templates.TemplateResponse(
        request,
        "fragments/documents_table.html",
        {"agent_id": agent_id, "docs": docs, "polling": has_inflight(docs)},
    )


@router.delete("/agents/{agent_id}/documents/{document_id}", response_class=HTMLResponse)
async def delete_document(
    agent_id: UUID, document_id: UUID, c: ContainerDep, owner: OwnerDep
) -> Response:
    """Пустой ответ: HTMX удаляет строку (hx-swap="delete")."""
    await c.documents.delete(owner, agent_id, document_id)
    return HTMLResponse("")
