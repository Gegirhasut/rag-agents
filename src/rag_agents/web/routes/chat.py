import html
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from rag_agents.domain.answers import (
    DoneEvent,
    ErrorEvent,
    SourcesEvent,
    StreamEvent,
    TokenEvent,
)
from rag_agents.services.errors import NotFoundError
from rag_agents.web.deps import ContainerDep, OwnerDep
from rag_agents.web.rendering import render_answer
from rag_agents.web.sse import format_sse, with_heartbeat
from rag_agents.web.templating import templates

router = APIRouter()
log = structlog.get_logger()
_MIN_QUESTION = 2
_MAX_QUESTION = 2000


@router.post("/agents/{agent_id}/chats/{chat_ref}/messages", response_class=HTMLResponse)
async def post_message(
    request: Request,
    agent_id: UUID,
    chat_ref: str,
    c: ContainerDep,
    owner: OwnerDep,
    question: Annotated[str, Form()] = "",
) -> Response:
    """chat_ref = "new" (последний или новый чат) или UUID чата (ARCHITECTURE §6.1)."""
    question = question.strip()
    if not _MIN_QUESTION <= len(question) <= _MAX_QUESTION:
        raise HTTPException(422, "Вопрос должен быть от 2 до 2000 символов")
    try:
        chat_id = None if chat_ref == "new" else UUID(chat_ref)
        pair = await c.query.ask(owner, agent_id, chat_id, question)
    except (NotFoundError, ValueError) as e:
        raise HTTPException(404, "Агент или чат не найден") from e
    return templates.TemplateResponse(
        request,
        "fragments/message_pair.html",
        {"agent_id": agent_id, "q": pair.question, "a": pair.answer},
    )


def _render_event(message_id: UUID, ev: StreamEvent) -> tuple[str, str]:
    """StreamEvent → (имя SSE-события, HTML-фрагмент для htmx-ext-sse)."""
    anchor = f"src-{message_id}"
    match ev:
        case TokenEvent():
            return "token", html.escape(ev.delta)
        case SourcesEvent():
            body = templates.get_template("fragments/sources.html").render(
                citations=ev.citations, anchor=anchor
            )
            return "sources", body
        case DoneEvent():
            r = ev.result
            body = templates.get_template("fragments/answer_final.html").render(
                answer_html=render_answer(r.answer_md, anchor, len(r.citations)),
                result=r,
                anchor=anchor,
            )
            return "done", body
        case ErrorEvent():
            body = templates.get_template("fragments/answer_error.html").render(error=ev)
            # sse-close слушает done: ошибку тоже отдаём событием done, чтобы закрыть EventSource
            return "done", body


@router.get("/agents/{agent_id}/messages/{message_id}/stream")
async def stream_message(
    request: Request, agent_id: UUID, message_id: UUID, c: ContainerDep, owner: OwnerDep
) -> Response:
    try:
        events = await c.query.stream_answer(owner, agent_id, message_id)
    except NotFoundError as e:
        raise HTTPException(404, "Сообщение не найдено") from e

    async def sse() -> AsyncIterator[str]:
        n = 0
        async for ev in events:
            name, data = _render_event(message_id, ev)
            n += 1
            yield format_sse(name, data, event_id=n)

    return StreamingResponse(
        with_heartbeat(sse()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
