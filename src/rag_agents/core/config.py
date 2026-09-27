from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ReasoningEffort = Literal["low", "high", "max"]


class Settings(BaseSettings):
    """Все настройки — из окружения (.env в compose). Значений-секретов по умолчанию нет."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = True

    database_url: str = "postgresql+asyncpg://rag:rag@postgres:5432/rag_agents"
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "amqp://rag:rag@rabbitmq:5672//"
    # Management API RabbitMQ (очереди для /system); логин и пароль берутся из celery_broker_url
    rabbitmq_management_url: str = "http://rabbitmq:15672"

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection_prefix: str = ""  # тесты изолируют коллекции префиксом

    ollama_base_url: str = "http://ollama:11434"
    embedding_model: str = "bge-m3:567m"
    embedding_dim: int = 1024
    embedding_batch_size: int = 32
    # Потоки llama.cpp в Ollama. По умолчанию Ollama берёт все ядра хоста и на cgroup-квоте
    # контейнера (cpus) упирается в троттлинг — задаём равным cpus контейнера ollama.
    embedding_num_thread: int | None = None
    tokenizer_path: Path = Path("/opt/models/bge-m3/tokenizer.json")

    # LLM: DeepSeek через OpenAI-совместимый API
    llm_base_url: str = "https://api.deepseek.com"
    deepseek_api_key: SecretStr | None = None
    llm_model: str = "deepseek-flash"
    # Пустое значение в .env = не передавать параметр провайдеру
    llm_reasoning_effort: ReasoningEffort | None = "low"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1200
    llm_connect_timeout_s: float = 5.0
    llm_read_timeout_s: float = 60.0

    upload_dir: Path = Path("/data/uploads")
    max_upload_mb: int = Field(100, ge=1)

    chunk_target_tokens: int = 400
    chunk_max_tokens: int = 512

    seed_user_email: str = "owner@local"

    # Живые события пайплайна для страницы /system (Redis pub/sub). В проде — выключить
    # или закрыть админ-доступом (итерация 2): на странице видны скоры и метаданные книг.
    trace_enabled: bool = True

    @field_validator(
        "llm_reasoning_effort", "deepseek_api_key", "embedding_num_thread", mode="before"
    )
    @classmethod
    def _empty_is_none(cls, v: object) -> object:
        return None if isinstance(v, str) and not v.strip() else v

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
