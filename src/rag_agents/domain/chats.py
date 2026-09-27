from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from rag_agents.domain.enums import MessageRole, MessageStatus


class ChatOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    agent_id: UUID
    user_id: UUID
    title: str | None
    created_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    chat_id: UUID
    role: MessageRole
    content: str
    status: MessageStatus
    citations: list[dict[str, Any]] | None
    refused: bool | None
    usage: dict[str, Any] | None
    trace_id: str | None = None
    feedback: int | None = None
    created_at: datetime


class MessagePair(BaseModel):
    question: MessageOut
    answer: MessageOut


class FeedbackStat(BaseModel):
    """Оценки ответов агента за период (👍 = up, 👎 = down)."""

    agent_id: UUID
    up: int
    down: int
