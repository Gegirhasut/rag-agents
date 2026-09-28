from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header, Response
from fastapi.responses import StreamingResponse

from rag_agents.api.v1.schemas import ERRORS
from rag_agents.domain.answers import QueryRequest, QueryResponse
from rag_agents.services.query import AnswerFailedError
from rag_agents.services.ratelimit import Rule
from rag_agents.web.deps import ContainerDep, OwnerDep
from rag_agents.web.errors import problem
from rag_agents.web.sse import format_sse, with_heartbeat

router = APIRouter(prefix="/agents/{agent_id}", tags=["query"], responses=ERRORS)  # type: ignore[arg-type]
SSE = "text/event-stream"


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        200: {
            "content": {SSE: {"schema": {"type": "string"}}},
            "description": "JSON QueryResponse или (stream=true / Accept: text/event-stream) "
            "SSE-события token, sources, done, error; data — JSON StreamEvent",
        },
        429: {"description": "Лимит вопросов в минуту"},
    },
)
async def query(
    agent_id: UUID,
    body: QueryRequest,
    c: ContainerDep,
    owner: OwnerDep,
    accept: Annotated[str, Header()] = "",
) -> Response:
    """Вопрос агенту. Без chat_id открывается новый чат; с chat_id — продолжение."""
    await c.rate_limiter.check(Rule("questions", c.settings.rl_questions_per_min, 60), str(owner))
    pair = await c.query.ask(owner, agent_id, body.chat_id, body.question, new_chat=True)
    ids = {"X-Chat-Id": str(pair.answer.chat_id), "X-Message-Id": str(pair.answer.id)}

    if body.stream or SSE in accept:
        events = await c.query.stream_answer(owner, agent_id, pair.answer.id)

        async def sse() -> AsyncIterator[str]:
            n = 0
            async for ev in events:
                n += 1
                yield format_sse(ev.type, ev.model_dump_json(), event_id=n)

        return StreamingResponse(
            with_heartbeat(sse()),
            media_type=SSE,
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **ids},
        )

    try:
        result = await c.query.answer(owner, agent_id, pair.answer.id)
    except AnswerFailedError as e:
        # LLM или retrieval недоступны: 503, если повтор имеет смысл
        return problem(
            503 if e.event.retryable else 502,
            e.event.message,
            code=e.event.code,
            headers=ids,
            retryable=e.event.retryable,
        )
    response = QueryResponse(chat_id=pair.answer.chat_id, message_id=pair.answer.id, result=result)
    return Response(response.model_dump_json(), media_type="application/json", headers=ids)
