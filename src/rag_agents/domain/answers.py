from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from rag_agents.domain.documents import ChunkPayload


class RetrievedChunk(BaseModel):
    chunk_id: UUID
    document_id: UUID
    score: float
    payload: ChunkPayload


class Citation(BaseModel):
    n: int
    chunk_id: UUID
    document_id: UUID
    book_title: str | None
    author: str | None
    chapter_title: str | None
    section_path: list[str]
    snippet: str
    score: float

    @property
    def label(self) -> str:
        parts = []
        if self.author:
            parts.append(self.author)
        if self.book_title:
            parts.append(f"«{self.book_title}»")
        head = " — ".join(parts) if parts else "Источник"
        if self.chapter_title:
            head += f", гл. {self.chapter_title}"
        return head


class AnswerUsage(BaseModel):
    provider: str
    model: str
    reasoning_effort: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    t_embed_ms: int | None = None  # None — ответы до появления поля
    t_search_ms: int | None = None
    t_retrieval_ms: int
    t_first_token_ms: int | None = None
    t_total_ms: int
    # Стоимость по configs/llm_prices.yaml с учётом тарифа peak/off-peak (None — цены нет)
    cost_usd: float | None = None
    cost_peak: bool | None = None


class QueryResult(BaseModel):
    answer_md: str
    refused: bool
    citations: list[Citation]
    usage: AnswerUsage | None
    trace_id: str | None = None  # трейс в Langfuse (страница /insights/traces/{id})


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    delta: str


class SourcesEvent(BaseModel):
    type: Literal["sources"] = "sources"
    citations: list[Citation]


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    result: QueryResult


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str
    retryable: bool


StreamEvent = Annotated[
    TokenEvent | SourcesEvent | DoneEvent | ErrorEvent, Field(discriminator="type")
]
