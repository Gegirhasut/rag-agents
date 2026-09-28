import re
from uuid import UUID, uuid5

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag_agents.core.errors import TransientError
from rag_agents.domain.answers import RetrievedChunk
from rag_agents.domain.documents import ChunkPayload

# Фиксированный namespace: id точки = uuid5(NS, "{document_id}:{chunking_version}:{ord}")
CHUNK_NAMESPACE = UUID("5b1f3f0e-6f0a-4a57-9d8e-3c2b8f0c7a11")
DENSE = "dense"
SPARSE = "bm25"


def chunk_point_id(document_id: UUID, chunking_version: int, ord_: int) -> UUID:
    return uuid5(CHUNK_NAMESPACE, f"{document_id}:{chunking_version}:{ord_}")


def collection_name(prefix: str, embedding_model: str, dim: int) -> str:
    """Коллекция на модель эмбеддингов (ARCHITECTURE §9.1): chunks__bge_m3_567m__1024."""
    model = re.sub(r"[^a-z0-9]+", "_", embedding_model.lower()).strip("_")
    return f"{prefix}chunks__{model}__{dim}"


def _agent_filter(agent_id: UUID, document_id: UUID | None = None) -> models.Filter:
    must: list[models.Condition] = [
        models.FieldCondition(key="agent_id", match=models.MatchValue(value=str(agent_id)))
    ]
    if document_id is not None:
        must.append(
            models.FieldCondition(
                key="document_id", match=models.MatchValue(value=str(document_id))
            )
        )
    return models.Filter(must=must)


class QdrantChunkIndex:
    """Доступ к векторам чанков. agent_id — обязательный аргумент каждой операции с данными.

    Метода поиска без фильтра агента нет намеренно (ARCHITECTURE §1, §9).
    """

    def __init__(self, client: AsyncQdrantClient, *, quantization: bool = False) -> None:
        self.client = client
        # int8-квантизация векторов в RAM. На CPU без AVX Qdrant 1.19 падает с SIGILL,
        # когда строит HNSW по квантизованным векторам — поэтому включается настройкой
        self.quantization = quantization

    def _quantization_config(self) -> models.ScalarQuantization | None:
        if not self.quantization:
            return None
        return models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
        )

    async def ensure_collection(self, name: str, dim: int) -> None:
        try:
            if await self.client.collection_exists(name):
                await self._sync_quantization(name)
                return
            await self.client.create_collection(
                name,
                vectors_config={
                    DENSE: models.VectorParams(
                        size=dim, distance=models.Distance.COSINE, on_disk=True
                    )
                },
                # Слот под BM25 (итерация 5): добавить sparse-вектор в существующую коллекцию нельзя
                sparse_vectors_config={
                    SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
                hnsw_config=models.HnswConfigDiff(m=0, payload_m=16),
                quantization_config=self._quantization_config(),
            )
            await self.client.create_payload_index(
                name,
                "agent_id",
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD, is_tenant=True
                ),
            )
            await self.client.create_payload_index(
                name, "document_id", field_schema=models.PayloadSchemaType.KEYWORD
            )
        except UnexpectedResponse as e:
            # Гонка двух воркеров: коллекция уже создана соседом
            if e.status_code == 409:  # noqa: PLR2004  HTTP Conflict
                return
            raise TransientError(f"qdrant: {e}") from e
        except ResponseHandlingException as e:
            raise TransientError(f"qdrant unreachable: {e}") from e

    async def _sync_quantization(self, name: str) -> None:
        """Коллекция создана с другой настройкой квантизации — приводим к текущей."""
        info = await self.client.get_collection(name)
        has = info.config.quantization_config is not None
        if has == self.quantization:
            return
        await self.client.update_collection(
            name, quantization_config=self._quantization_config() or models.Disabled.DISABLED
        )

    async def upsert(
        self,
        collection: str,
        agent_id: UUID,
        points: list[tuple[UUID, list[float], ChunkPayload]],
    ) -> None:
        for _, _, payload in points:
            if payload.agent_id != str(agent_id):
                raise ValueError("payload agent_id mismatch")
        try:
            await self.client.upsert(
                collection,
                points=[
                    models.PointStruct(id=str(pid), vector={DENSE: vec}, payload=p.model_dump())
                    for pid, vec, p in points
                ],
                wait=True,
            )
        except (ResponseHandlingException, UnexpectedResponse) as e:
            raise TransientError(f"qdrant upsert failed: {e}") from e

    async def search_dense(
        self, collection: str, agent_id: UUID, vector: list[float], limit: int
    ) -> list[RetrievedChunk]:
        try:
            resp = await self.client.query_points(
                collection,
                query=vector,
                using=DENSE,
                query_filter=_agent_filter(agent_id),
                limit=limit,
                with_payload=True,
            )
        except (ResponseHandlingException, UnexpectedResponse) as e:
            raise TransientError(f"qdrant search failed: {e}") from e
        out = []
        for p in resp.points:
            payload = ChunkPayload.model_validate(p.payload)
            out.append(
                RetrievedChunk(
                    chunk_id=UUID(str(p.id)),
                    document_id=UUID(payload.document_id),
                    score=p.score,
                    payload=payload,
                )
            )
        return out

    async def delete_document(self, collection: str, agent_id: UUID, document_id: UUID) -> None:
        try:
            await self.client.delete(
                collection,
                points_selector=models.FilterSelector(filter=_agent_filter(agent_id, document_id)),
                wait=True,
            )
        except (ResponseHandlingException, UnexpectedResponse) as e:
            raise TransientError(f"qdrant delete failed: {e}") from e

    async def delete_agent(self, collection: str, agent_id: UUID) -> None:
        """Все точки агента в коллекции (очистка удалённого агента)."""
        try:
            await self.client.delete(
                collection,
                points_selector=models.FilterSelector(filter=_agent_filter(agent_id)),
                wait=True,
            )
        except UnexpectedResponse as e:
            if e.status_code == 404:  # noqa: PLR2004  коллекции нет — нечего удалять
                return
            raise TransientError(f"qdrant delete failed: {e}") from e
        except ResponseHandlingException as e:
            raise TransientError(f"qdrant delete failed: {e}") from e

    # --- Чтение для страницы «Под капотом» (/system) ---

    async def list_collections(self) -> list[str]:
        resp = await self.client.get_collections()
        return [c.name for c in resp.collections]

    async def collection_info(self, name: str) -> models.CollectionInfo:
        """Инфраструктурные метаданные коллекции (без данных агентов)."""
        return await self.client.get_collection(name)

    async def count(self, collection: str, agent_id: UUID) -> int:
        resp = await self.client.count(collection, count_filter=_agent_filter(agent_id), exact=True)
        return resp.count

    async def scroll_vectors(
        self, collection: str, agent_id: UUID, limit: int
    ) -> list[tuple[str, list[float], ChunkPayload]]:
        """Все (до limit) точки агента вместе с dense-векторами — для 2D-карты."""
        out: list[tuple[str, list[float], ChunkPayload]] = []
        offset: models.ExtendedPointId | None = None
        while len(out) < limit:
            points, offset = await self.client.scroll(
                collection,
                scroll_filter=_agent_filter(agent_id),
                limit=min(256, limit - len(out)),
                offset=offset,
                with_payload=True,
                with_vectors=[DENSE],
            )
            for p in points:
                vec = p.vector.get(DENSE) if isinstance(p.vector, dict) else None
                if isinstance(vec, list) and vec and isinstance(vec[0], float):
                    out.append((str(p.id), vec, ChunkPayload.model_validate(p.payload)))
            if offset is None:
                break
        return out

    async def get_point(
        self, collection: str, agent_id: UUID, point_id: UUID
    ) -> tuple[list[float], ChunkPayload] | None:
        """Одна точка с вектором; чужая (другой agent_id) — как несуществующая."""
        found = await self.client.retrieve(
            collection, ids=[str(point_id)], with_payload=True, with_vectors=[DENSE]
        )
        if not found:
            return None
        payload = ChunkPayload.model_validate(found[0].payload)
        vec = found[0].vector.get(DENSE) if isinstance(found[0].vector, dict) else None
        if payload.agent_id != str(agent_id) or not isinstance(vec, list):
            return None
        return [float(v) for v in vec if isinstance(v, float)], payload
