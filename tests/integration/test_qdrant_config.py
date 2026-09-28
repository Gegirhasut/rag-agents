"""Конфигурация коллекции: int8-квантизация выключается настройкой (VM без AVX).

На CPU без AVX Qdrant 1.19 падает с SIGILL, когда строит HNSW по квантизованным векторам
(SPEC §9, «что пошло не так»). Коллекция, созданная с квантизацией, переводится без неё.
"""

import pytest
from qdrant_client import AsyncQdrantClient, models

from rag_agents.rag.index.qdrant import QdrantChunkIndex, collection_name

pytestmark = pytest.mark.integration
DIM = 4


async def test_collection_without_quantization_by_default(
    qdrant: AsyncQdrantClient, collection_prefix: str
) -> None:
    name = collection_name(collection_prefix, "m", DIM)
    await QdrantChunkIndex(qdrant).ensure_collection(name, DIM)
    info = await qdrant.get_collection(name)
    assert info.config.quantization_config is None
    await qdrant.delete_collection(name)


async def test_existing_quantized_collection_is_switched_off(
    qdrant: AsyncQdrantClient, collection_prefix: str
) -> None:
    name = collection_name(collection_prefix, "m", DIM)
    await QdrantChunkIndex(qdrant, quantization=True).ensure_collection(name, DIM)
    info = await qdrant.get_collection(name)
    assert isinstance(info.config.quantization_config, models.ScalarQuantization)

    await QdrantChunkIndex(qdrant, quantization=False).ensure_collection(name, DIM)
    info = await qdrant.get_collection(name)
    assert info.config.quantization_config is None
    await qdrant.delete_collection(name)
