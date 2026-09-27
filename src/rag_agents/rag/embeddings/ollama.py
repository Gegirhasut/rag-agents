from typing import Protocol

import httpx

from rag_agents.core.errors import PermanentError, TransientError

_HTTP_SERVER_ERROR = 500


class Embedder(Protocol):
    model: str
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    """Dense-эмбеддинги через Ollama /api/embed (батчем). Векторы уже L2-нормализованы."""

    def __init__(
        self, client: httpx.AsyncClient, model: str, dim: int, num_thread: int | None = None
    ) -> None:
        self.client = client
        self.model = model
        self.dim = dim
        self.num_thread = num_thread

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        body: dict[str, object] = {"model": self.model, "input": texts, "truncate": True}
        if self.num_thread:
            body["options"] = {"num_thread": self.num_thread}
        try:
            resp = await self.client.post("/api/embed", json=body)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            raise TransientError(f"ollama unreachable: {e!r}") from e
        if resp.status_code >= _HTTP_SERVER_ERROR:
            raise TransientError(f"ollama {resp.status_code}: {resp.text[:200]}")
        if resp.is_error:
            raise PermanentError(
                "embedding_failed", f"Ollama {resp.status_code}: {resp.text[:200]}"
            )
        vectors: list[list[float]] = resp.json()["embeddings"]
        if len(vectors) != len(texts) or any(len(v) != self.dim for v in vectors):
            raise PermanentError("embedding_failed", "Ollama вернула векторы неожиданной формы")
        return vectors
