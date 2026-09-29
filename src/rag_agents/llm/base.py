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
    # response_format=json_object (OpenAI-совместимые API): ответ — один JSON-объект
    json_mode: bool = False


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


class Completion(BaseModel):
    text: str
    usage: LLMUsage
    # "length" у reasoning-моделей часто значит, что весь max_tokens ушёл на рассуждения
    finish_reason: str | None = None


async def complete(llm: LLMProvider, req: LLMRequest) -> Completion:
    """Ответ целиком (без стрима наружу): судья eval, в будущем condense."""
    parts: list[str] = []
    usage = LLMUsage()
    finish: str | None = None
    async for chunk in llm.stream(req):
        parts.append(chunk.delta)
        if chunk.usage:
            usage = chunk.usage
        if chunk.finish_reason:
            finish = chunk.finish_reason
    return Completion(text="".join(parts), usage=usage, finish_reason=finish)
