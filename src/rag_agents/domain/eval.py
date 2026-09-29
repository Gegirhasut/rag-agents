"""Контракты eval (ARCHITECTURE §15): golden-датасет, результат вопроса, прогон."""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_agents.domain.answers import QueryResult, RetrievedChunk


class EvalCategory(StrEnum):
    FACTUAL = "factual"
    INTERPRETIVE = "interpretive"
    MULTI_HOP = "multi_hop"
    OUT_OF_CORPUS = "out_of_corpus"  # ожидается отказ
    INJECTION = "injection"  # вопрос с попыткой переписать правила; ожидается отказ

    @property
    def expects_refusal(self) -> bool:
        return self in {EvalCategory.OUT_OF_CORPUS, EvalCategory.INJECTION}


class ExpectedSource(BaseModel):
    """Где лежит ответ. document — подстрока названия книги; section — путь внутри книги
    (например ["ЧАСТЬ ПЕРВАЯ", "I"]), совпадение — подпоследовательность section_path чанка.
    Без section совпадение по документу."""

    model_config = ConfigDict(frozen=True)
    document: str
    section: list[str] = Field(default_factory=list)


class GoldenItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    category: EvalCategory
    question: str
    reference_answer: str = ""
    key_facts: list[str] = Field(default_factory=list)
    expected_sources: list[ExpectedSource] = Field(default_factory=list)
    # Строки, которых не должно быть в ответе (маркеры выполненной инъекции)
    forbidden: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "GoldenItem":
        if self.category.expects_refusal:
            if self.expected_sources:
                raise ValueError(f"{self.id}: у вопроса с ожидаемым отказом нет источников")
        elif not (self.expected_sources and self.reference_answer):
            raise ValueError(f"{self.id}: нужны expected_sources и reference_answer")
        return self


class EvalAnswer(BaseModel):
    """Результат QueryService.evaluate: ответ (или код ошибки) и все найденные кандидаты."""

    result: QueryResult | None
    error: str | None
    retrieved: list[RetrievedChunk]  # search_k кандидатов по убыванию score
    context_k: int  # сколько из них ушло в промпт
    trace_id: str | None = None


class RetrievedRef(BaseModel):
    """Найденный чанк в eval_items.retrieved: без текста, только координаты и score."""

    chunk_id: UUID
    book_title: str | None
    section_path: list[str]
    score: float
    relevant: bool  # совпал с expected_sources


class EvalItemResult(BaseModel):
    item_id: str
    question: str
    category: EvalCategory
    answer: str | None
    error: str | None = None
    refused: bool | None
    retrieved: list[RetrievedRef]
    # Все метрики вопроса плоским словарём: hit@5, mrr@10, faithfulness, … None — не определена
    scores: dict[str, float | None]
    judge: dict[str, Any] | None = None  # сырые вердикты судьи (для разбора худших примеров)
    trace_id: str | None = None
    latency_ms: int | None
    ttft_ms: int | None
    cost_usd: float | None


class EvalRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    agent_id: UUID
    dataset: str
    dataset_sha: str
    config_name: str
    config: dict[str, Any]
    metrics: dict[str, Any] | None
    started_at: datetime
    finished_at: datetime | None
