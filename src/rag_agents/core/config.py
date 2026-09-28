from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
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
    # Таблица цен LLM (peak/off-peak): cost_usd считаем сами, Langfuse получает готовый cost
    llm_prices_path: Path = Path("configs/llm_prices.yaml")

    upload_dir: Path = Path("/data/uploads")
    max_upload_mb: int = Field(100, ge=1)

    chunk_target_tokens: int = 400
    chunk_max_tokens: int = 512

    seed_user_email: str = "owner@local"

    # Auth (ARCHITECTURE ADR-2): серверные сессии в Redis, cookie без данных — только id
    session_cookie: str = "rag_sid"
    session_ttl_s: int = 7 * 24 * 3600
    # Secure-cookie только за TLS; в dev UI открывается по http://192.168.56.10:8080
    session_cookie_secure: bool = False

    # Rate limit (FR-6.3), sliding window в Redis
    rate_limit_enabled: bool = True
    rl_questions_per_min: int = 20
    rl_uploads_per_hour: int = 30
    rl_login_per_min: int = 5

    # Живые события пайплайна для страницы /system (Redis pub/sub). Страница доступна только
    # администраторам: в событиях видны скоры и метаданные книг всех пользователей.
    trace_enabled: bool = True

    # Langfuse Cloud (EU). Без обоих ключей трейсинг выключен (no-op). LANGFUSE_HOST — старое
    # имя переменной у Langfuse SDK, поддерживаем как fallback.
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = Field(
        "https://cloud.langfuse.com",
        validation_alias=AliasChoices("LANGFUSE_BASE_URL", "LANGFUSE_HOST"),
    )
    langfuse_enabled: bool = True

    @field_validator(
        "llm_reasoning_effort",
        "deepseek_api_key",
        "embedding_num_thread",
        "langfuse_public_key",
        "langfuse_secret_key",
        mode="before",
    )
    @classmethod
    def _empty_is_none(cls, v: object) -> object:
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("langfuse_base_url", mode="before")
    @classmethod
    def _empty_base_url_is_cloud_eu(cls, v: object) -> object:
        return "https://cloud.langfuse.com" if isinstance(v, str) and not v.strip() else v

    @property
    def langfuse_active(self) -> bool:
        """Тесты никогда не шлют трейсы, даже если ключи лежат в локальном .env."""
        return (
            self.langfuse_enabled
            and self.app_env != "test"
            and self.langfuse_public_key is not None
            and self.langfuse_secret_key is not None
        )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
