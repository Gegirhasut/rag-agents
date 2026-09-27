from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agents.core.ids import uuid7
from rag_agents.domain.chats import ChatOut, MessageOut
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
            )
        )
