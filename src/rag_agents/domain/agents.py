from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class RetrievalSettings(BaseModel):
    """Подмножество ARCHITECTURE §5, нужное итерации 1. Остальные поля добавятся в итерации 5."""

    model_config = ConfigDict(frozen=True)
    top_k: int = Field(6, ge=1, le=20)


class GenerationSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    temperature: float = Field(0.3, ge=0, le=1.5)
    max_output_tokens: int = Field(1200, ge=100, le=4000)


class AgentSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    retrieval: RetrievalSettings = RetrievalSettings()
    generation: GenerationSettings = GenerationSettings()


class AgentCreate(BaseModel):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
    description: Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)] = ""
    persona_prompt: (
        Annotated[str, StringConstraints(strip_whitespace=True, max_length=4000)] | None
    ) = None


class AgentUpdate(BaseModel):
    """PATCH: переданные поля меняются, отсутствующие — нет (model_fields_set)."""

    name: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
        | None
    ) = None
    description: (
        Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)] | None
    ) = None
    persona_prompt: (
        Annotated[str, StringConstraints(strip_whitespace=True, max_length=4000)] | None
    ) = None


class AgentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    owner_id: UUID
    name: str
    slug: str
    description: str
    persona_prompt: str | None
    settings: AgentSettings
    active_index_id: UUID | None
    corpus_version: int
    created_at: datetime


class AgentIndexOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    agent_id: UUID
    embedding_model: str
    dim: int
    collection: str
    chunking_version: int


class AgentListItem(BaseModel):
    agent: AgentOut
    documents_total: int
    documents_done: int
