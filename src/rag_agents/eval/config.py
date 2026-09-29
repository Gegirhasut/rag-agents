from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rag_agents.domain.agents import RetrievalSettings


class EvalConfig(BaseModel):
    """configs/eval/<name>.yaml: что меняется в эксперименте относительно настроек агента."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str = ""
    # None — настройки агента как есть; иначе перекрывают agent.settings.retrieval
    retrieval: RetrievalSettings | None = None
    search_k: int = Field(20, ge=1, le=100)  # кандидатов для hit@k/recall@k
    judge: bool = True
    concurrency: int = Field(2, ge=1, le=8)
    langfuse: bool = True  # выгрузка в Langfuse Datasets (если ключи заданы)


def load_config(path: Path) -> EvalConfig:
    data: dict[str, Any] = yaml.safe_load(path.read_text("utf-8")) or {}
    data.setdefault("name", path.stem)
    return EvalConfig.model_validate(data)
