from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agents.domain.answers import AnswerUsage
from rag_agents.domain.enums import MessageRole, MessageStatus
from rag_agents.domain.insights import AnswerFact, IngestFact
from rag_agents.models.entities import Agent, Chat, Document, Message

# Страница «Аналитика» агрегирует в Python: на масштабе pet-проекта это тысячи строк.
# При росте — перенести перцентили в SQL (percentile_cont) и добавить индекс по created_at.
MAX_ROWS = 20_000


class InsightsRepository:
    """Сырьё для сводки «Аналитики». Только данные владельца (user_id / owner_id)."""

    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def answers(self, user_id: UUID, since: datetime) -> list[AnswerFact]:
        rows = await self.s.execute(
            select(Message, Chat.agent_id, Agent.name)
            .join(Chat, Chat.id == Message.chat_id)
            .join(Agent, Agent.id == Chat.agent_id)
            .where(
                Chat.user_id == user_id,
                Message.role == MessageRole.ASSISTANT,
                Message.status.in_([MessageStatus.DONE, MessageStatus.ERROR]),
                Message.created_at >= since,
            )
            .order_by(Message.created_at.desc())
            .limit(MAX_ROWS)
        )
        return [
            AnswerFact(
                message_id=m.id,
                chat_id=m.chat_id,
                agent_id=agent_id,
                agent_name=name,
                created_at=m.created_at,
                status=m.status.value,
                question=m.standalone_question,
                usage=AnswerUsage.model_validate(m.usage) if m.usage else None,
                refused=m.refused,
                feedback=m.feedback,
                trace_id=m.trace_id,
            )
            for m, agent_id, name in rows.all()
        ]

    async def documents(self, owner_id: UUID, since: datetime) -> list[IngestFact]:
        rows = await self.s.execute(
            select(Document, Agent.name)
            .join(Agent, Agent.id == Document.agent_id)
            .where(Agent.owner_id == owner_id, Document.created_at >= since)
            .order_by(Document.created_at.desc())
            .limit(MAX_ROWS)
        )
        return [
            IngestFact(
                document_id=d.id,
                agent_id=d.agent_id,
                agent_name=name,
                filename=d.filename,
                status=d.status.value,
                created_at=d.created_at,
                started_at=d.started_at,
                finished_at=d.finished_at,
                chunks_total=d.chunks_total,
                error_message=d.error_message,
                timings={k: int(v) for k, v in (d.meta or {}).get("timings", {}).items()},
            )
            for d, name in rows.all()
        ]
