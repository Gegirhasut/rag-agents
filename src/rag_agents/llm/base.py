from collections.abc import AsyncIterator
from typing import Literal, Protocol

from pydantic import BaseModel


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None


class LLMRequest(BaseModel):
    messages: list[LLMMessage]
    temperature: float = 0.3
    max_tokens: int = 1200
    purpose: Literal["answer", "condense", "judge", "agent_step"] = "answer"


class LLMUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0


class LLMChunk(BaseModel):
    """Элемент стрима. Рассуждения модели (reasoning) наружу не отдаются — только в usage."""

    delta: str = ""
    finish_reason: str | None = None
    usage: LLMUsage | None = None


class LLMError(Exception):
    def __init__(self, message: str, *, retryable: bool, status: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class LLMProvider(Protocol):
    name: str
    model: str
    reasoning_effort: str | None

    def stream(self, req: LLMRequest) -> AsyncIterator[LLMChunk]: ...
