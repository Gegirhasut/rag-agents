from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agents.core.ids import uuid7
from rag_agents.domain.chats import ChatOut, FeedbackStat, MessageOut
from rag_agents.domain.enums import MessageRole, MessageStatus
from rag_agents.models.entities import Chat, Message


class ChatRepository:
    """Чаты и сообщения. Доступ к сообщению всегда через (agent_id, user_id) его чата."""

    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def latest_chat(self, agent_id: UUID, user_id: UUID) -> ChatOut | None:
        chat = await self.s.scalar(
            select(Chat)
            .where(Chat.agent_id == agent_id, Chat.user_id == user_id)
            .order_by(Chat.created_at.desc(), Chat.id.desc())
            .limit(1)
        )
        return ChatOut.model_validate(chat) if chat else None

    async def get_chat(self, agent_id: UUID, user_id: UUID, chat_id: UUID) -> ChatOut | None:
        chat = await self.s.scalar(
            select(Chat).where(
                Chat.id == chat_id, Chat.agent_id == agent_id, Chat.user_id == user_id
            )
        )
        return ChatOut.model_validate(chat) if chat else None

    async def create_chat(self, agent_id: UUID, user_id: UUID, title: str | None) -> ChatOut:
        chat = Chat(id=uuid7(), agent_id=agent_id, user_id=user_id, title=title)
        self.s.add(chat)
        await self.s.flush()
        await self.s.refresh(chat)
        return ChatOut.model_validate(chat)

    async def add_message(
        self,
        chat_id: UUID,
        role: MessageRole,
        status: MessageStatus,
        content: str = "",
        standalone_question: str | None = None,
        index_id: UUID | None = None,
    ) -> MessageOut:
        msg = Message(
            id=uuid7(),
            chat_id=chat_id,
            role=role,
            status=status,
            content=content,
            standalone_question=standalone_question,
            index_id=index_id,
        )
        self.s.add(msg)
        await self.s.flush()
        await self.s.refresh(msg)
        return MessageOut.model_validate(msg)

    async def list_messages(self, chat_id: UUID) -> list[MessageOut]:
        rows = await self.s.scalars(
            select(Message).where(Message.chat_id == chat_id).order_by(Message.id)
        )
        return [MessageOut.model_validate(m) for m in rows]

    async def get_message(
        self, agent_id: UUID, user_id: UUID, message_id: UUID
    ) -> tuple[MessageOut, str | None] | None:
        """Сообщение + вопрос, на который оно отвечает (standalone_question)."""
        row = await self.s.execute(
            select(Message)
            .join(Chat, Chat.id == Message.chat_id)
            .where(Message.id == message_id, Chat.agent_id == agent_id, Chat.user_id == user_id)
        )
        msg = row.scalar_one_or_none()
        if msg is None:
            return None
        return MessageOut.model_validate(msg), msg.standalone_question

    async def claim_for_streaming(self, message_id: UUID) -> bool:
        result = await self.s.execute(
            update(Message)
            .where(Message.id == message_id, Message.status == MessageStatus.PENDING)
            .values(status=MessageStatus.STREAMING)
            .returning(Message.id)
        )
        return result.scalar_one_or_none() is not None

    async def finish_message(
        self,
        message_id: UUID,
        *,
        status: MessageStatus,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        refused: bool | None = None,
        usage: dict[str, Any] | None = None,
        prompt_version: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        await self.s.execute(
            update(Message)
            .where(Message.id == message_id)
            .values(
                status=status,
                content=content,
                citations=citations,
                refused=refused,
                # grounded (валидация [n]) появится в итерации 5 — до неё не заполняем
                usage=usage,
                prompt_version=prompt_version,
                trace_id=trace_id,
            )
        )

    async def set_feedback(
        self, agent_id: UUID, user_id: UUID, message_id: UUID, value: int
    ) -> MessageOut | None:
        """Оценка только завершённого ответа ассистента в чате этого агента и пользователя."""
        in_scope = (
            select(Message.id)
            .join(Chat, Chat.id == Message.chat_id)
            .where(
                Message.id == message_id,
                Message.role == MessageRole.ASSISTANT,
                Message.status == MessageStatus.DONE,
                Chat.agent_id == agent_id,
                Chat.user_id == user_id,
            )
        )
        msg = await self.s.scalar(
            update(Message)
            .where(Message.id.in_(in_scope))
            .values(feedback=value)
            .returning(Message)
        )
        return MessageOut.model_validate(msg) if msg else None

    async def feedback_stats(self, user_id: UUID, since: datetime) -> list[FeedbackStat]:
        """👍/👎 по агентам пользователя за период (по времени ответа)."""
        rows = await self.s.execute(
            select(
                Chat.agent_id,
                func.count().filter(Message.feedback == 1),
                func.count().filter(Message.feedback == -1),
            )
            .join(Chat, Chat.id == Message.chat_id)
            .where(
                Chat.user_id == user_id,
                Message.feedback.is_not(None),
                Message.created_at >= since,
            )
            .group_by(Chat.agent_id)
        )
        return [FeedbackStat(agent_id=a, up=up, down=down) for a, up, down in rows.all()]

    async def feedback_by_trace(self, user_id: UUID, trace_ids: list[str]) -> dict[str, int]:
        """trace_id → оценка (1 / -1) для ответов этого пользователя."""
        if not trace_ids:
            return {}
        rows = await self.s.execute(
            select(Message.trace_id, Message.feedback)
            .join(Chat, Chat.id == Message.chat_id)
            .where(
                Chat.user_id == user_id,
                Message.trace_id.in_(trace_ids),
                Message.feedback.is_not(None),
            )
        )
        return {str(t): int(f) for t, f in rows.all()}
