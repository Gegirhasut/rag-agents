import json

import httpx
import pytest

from rag_agents.core.errors import PermanentError, TransientError
from rag_agents.rag.embeddings.ollama import OllamaEmbedder


def embedder(handler: httpx.MockTransport, num_thread: int | None = 3) -> OllamaEmbedder:
    client = httpx.AsyncClient(base_url="http://ollama", transport=handler)
    return OllamaEmbedder(client, "bge-m3:567m", dim=2, num_thread=num_thread)


async def test_sends_batch_and_thread_option() -> None:
    sent: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    vectors = await embedder(httpx.MockTransport(handle)).embed(["а", "б"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert sent[0] == {
        "model": "bge-m3:567m",
        "input": ["а", "б"],
        "truncate": True,
        "options": {"num_thread": 3},
    }


async def test_no_thread_option_when_not_configured() -> None:
    sent: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})

    await embedder(httpx.MockTransport(handle), num_thread=None).embed(["а"])
    assert "options" not in sent[0]


async def test_server_error_is_transient_and_bad_shape_is_permanent() -> None:
    with pytest.raises(TransientError):
        await embedder(httpx.MockTransport(lambda _: httpx.Response(503))).embed(["а"])
    bad = httpx.MockTransport(lambda _: httpx.Response(200, json={"embeddings": [[0.1]]}))
    with pytest.raises(PermanentError):
        await embedder(bad).embed(["а"])
