"""Снимок состояния инфраструктуры и карта векторов для страницы «Под капотом» (/system).

Каждый источник опрашивается независимо и с таймаутом: недоступный сервис даёт
карточку с ошибкой, а не падение страницы.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import anyio
import httpx
import numpy as np
import structlog
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agents.core.config import Settings
from rag_agents.core.db import Database
from rag_agents.domain.agents import AgentOut
from rag_agents.domain.system import (
    AdminUi,
    CeleryStats,
    LlmInfo,
    OllamaModel,
    OllamaStats,
    PointDetail,
    PostgresStats,
    QdrantCollectionStats,
    QdrantStats,
    QueueStats,
    RedisKey,
    RedisStats,
    SystemSnapshot,
    VectorMap,
    VectorMapDoc,
    VectorPoint,
    WorkerTask,
)
from rag_agents.domain.tasks import BuildVectorMapTask
from rag_agents.rag.index.qdrant import QdrantChunkIndex
from rag_agents.repositories.agents import AgentRepository
from rag_agents.repositories.system import PgStatsRepository
from rag_agents.services.errors import NotFoundError
from rag_agents.services.publisher import TaskPublisher

log = structlog.get_logger()

# Any: сырой ответ celery inspect (dict произвольной формы)
WorkerInspector = Callable[[], dict[str, list[dict[str, Any]]]]

_PROBE_TIMEOUT_S = 2.5
_SNAPSHOT_TIMEOUT_S = 4.0  # на VM под нагрузкой ответы источников гуляют до ~3 с
_MAP_LIMIT = 5000
# Карта зависит только от корпуса агента: ключ меняется вместе с corpus_version и числом точек
_MAP_CACHE_TTL_S = 7 * 24 * 3600
# Сколько ждать воркер, прежде чем поставить задачу заново (он мог упасть или быть занят книгой)
_MAP_LOCK_S = 600


@dataclass(frozen=True)
class _MapTarget:
    agent_id: UUID
    corpus_version: int
    total: int  # точек агента в коллекции
    collection: str
    dim: int

    @property
    def redis_key(self) -> str:
        return f"vmap:{self.agent_id}:{self.corpus_version}:{self.total}:{self.collection}"


_REDIS_SAMPLE = 25


@dataclass(frozen=True)
class AdminUiSpec:
    """Веб-админка из compose-профиля debug (или встроенная в сервис). Порт — как в compose.yaml."""

    key: str
    title: str
    probe_url: str  # адрес внутри docker-сети для проверки «запущено ли»
    port: int
    path: str
    about: str


ADMIN_UIS = (
    AdminUiSpec(
        "qdrant",
        "Qdrant Dashboard",
        "http://qdrant:6333/dashboard",
        6333,
        "/dashboard",
        "Встроен в Qdrant: коллекции, точки с payload и векторами, консоль REST-запросов, "
        "визуализация векторов.",
    ),
    AdminUiSpec(
        "rabbitmq",
        "RabbitMQ Management",
        "http://rabbitmq:15672/",
        15672,
        "/",
        "Встроен в образ rabbitmq:*-management: очереди, DLQ, соединения, скорость сообщений. "
        "Логин и пароль — RABBITMQ_DEFAULT_* из .env.",
    ),
    AdminUiSpec(
        "flower",
        "Flower",
        "http://flower:5555/",
        5555,
        "/",
        "Мониторинг Celery (аналог Laravel Horizon): воркеры, задачи, ретраи, время выполнения.",
    ),
    AdminUiSpec(
        "pgweb",
        "pgweb",
        "http://pgweb:8081/",
        8081,
        "/",
        "Лёгкий веб-клиент PostgreSQL: таблицы, строки, SQL-запросы (как Adminer/phpMyAdmin).",
    ),
    AdminUiSpec(
        "redisinsight",
        "RedisInsight",
        "http://redisinsight:5540/",
        5540,
        "/",
        "GUI для Redis: ключи, их типы и TTL, pub/sub-канал trace:events, профайлер команд. "
        "При первом открытии примите соглашение — подключение rag-agents добавится само.",
    ),
)


def pca_2d(
    vectors: np.ndarray, iters: int = 6, oversample: int = 6
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """PCA на 2 оси: (координаты n×2, среднее, компоненты 2×d, доля дисперсии по осям).

    Метод главных компонент находит две оси, вдоль которых векторы «разбросаны» сильнее всего.
    Проекция 1024 → 2 теряет почти всю информацию, но близкие по смыслу чанки остаются рядом.

    Полный SVD на CPU без AVX (наша VM) занимает секунды, поэтому считаем только нужные
    компоненты рандомизированным блочным методом (Halko и др.): ищем сразу 2 + oversample осей.
    Запас осей нужен потому, что у эмбеддингов первые компоненты почти равны (5.0 % и 4.5 %),
    и итерации по ровно двум осям сходятся десятками шагов. С запасом хватает нескольких.
    Итерации — в float32 (вдвое быстрее на CPU без AVX), координаты — в float64 от найденного
    базиса. На 3 000 × 1024 это ~0.7 с вместо ~8 с у 60 итераций по двум осям.
    """
    x = vectors.astype(np.float64)
    n, d = x.shape
    mean = x.mean(axis=0) if n else np.zeros(d)
    if n < 2:  # noqa: PLR2004  одной точке нечего раскладывать
        return np.zeros((n, 2)), mean, np.zeros((2, d)), [0.0, 0.0]
    centered = x - mean
    fast = centered.astype(np.float32)
    k = min(2 + oversample, n, d)
    probe = np.random.default_rng(0).normal(size=(n, k)).astype(np.float32)
    basis, _ = np.linalg.qr(fast.T @ probe)
    for _ in range(iters):
        basis, _ = np.linalg.qr(fast.T @ (fast @ basis))
    # Rayleigh–Ritz: внутри найденного подпространства берём две оси с наибольшей дисперсией
    sub = centered @ basis.astype(np.float64)
    w, v = np.linalg.eigh(sub.T @ sub)
    order = np.argsort(w)[::-1][:2]
    top = basis.astype(np.float64) @ v[:, order]
    top, _ = np.linalg.qr(top)  # ортонормируем после float32
    if top.shape[1] < 2:  # noqa: PLR2004  d == 1: второй оси нет
        top = np.hstack([top, np.zeros((d, 2 - top.shape[1]))])
    coords = centered @ top
    total = float((centered**2).sum()) or 1.0
    explained = [float(val / total) for val in (coords**2).sum(axis=0)]
    return coords, mean, top.T, explained


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:  # noqa: PLR2004
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


async def _guard[S](make: Callable[[], Awaitable[S]], fallback: Callable[[str], S]) -> S:
    try:
        return await asyncio.wait_for(make(), _SNAPSHOT_TIMEOUT_S)
    except Exception as e:  # любой сбой источника → карточка с ошибкой, не 500
        return fallback(f"{type(e).__name__}: {e}"[:300])


class SystemService:
    def __init__(
        self,
        db: Database,
        redis: Redis,
        index: QdrantChunkIndex,
        ollama_http: httpx.AsyncClient,
        rabbit_http: httpx.AsyncClient,
        inspect_workers: WorkerInspector,
        settings: Settings,
        publisher: TaskPublisher,
    ) -> None:
        self.db = db
        self.redis = redis
        self.publisher = publisher
        self.index = index
        self.ollama_http = ollama_http
        self.rabbit_http = rabbit_http
        self.inspect_workers = inspect_workers
        self.settings = settings
        self._maps: dict[UUID, tuple[str, VectorMap]] = {}
        self._probe_http = httpx.AsyncClient(timeout=1.5)

    async def aclose(self) -> None:
        await self._probe_http.aclose()

    async def snapshot(self, owner_id: UUID) -> SystemSnapshot:
        """Состояние всех сервисов параллельно; каждый источник — со своим таймаутом."""
        qdrant, celery, redis, pg, ollama, uis = await asyncio.gather(
            _guard(lambda: self._qdrant(owner_id), lambda e: QdrantStats(ok=False, error=e)),
            _guard(self._celery, lambda e: CeleryStats(ok=False, error=e)),
            _guard(self._redis, lambda e: RedisStats(ok=False, error=e)),
            _guard(self._postgres, lambda e: PostgresStats(ok=False, error=e)),
            _guard(
                self._ollama,
                lambda e: OllamaStats(ok=False, error=e, base_url=self.settings.ollama_base_url),
            ),
            self._admin_uis(),
        )
        s = self.settings
        return SystemSnapshot(
            qdrant=qdrant,
            celery=celery,
            redis=redis,
            postgres=pg,
            ollama=ollama,
            llm=LlmInfo(
                base_url=s.llm_base_url,
                model=s.llm_model,
                reasoning_effort=s.llm_reasoning_effort,
                key_configured=s.deepseek_api_key is not None,
            ),
            admin_uis=uis,
        )

    async def _qdrant(self, owner_id: UUID) -> QdrantStats:
        async with self.db.session() as s:
            agents = await AgentRepository(s).list_for_owner(owner_id)
        out = []
        for name in await self.index.list_collections():
            info = await self.index.collection_info(name)
            params = info.config.params
            vectors = []
            if isinstance(params.vectors, dict):
                for vname, v in params.vectors.items():
                    disk = ", on_disk" if v.on_disk else ""
                    vectors.append(f"{vname}: {v.size} × {v.distance.value}{disk}")
            sparse = [
                f"{k}: sparse, modifier={v.modifier.value if v.modifier else 'none'}"
                for k, v in (params.sparse_vectors or {}).items()
            ]
            h = info.config.hnsw_config
            quant = info.config.quantization_config
            quant_s = None
            if quant is not None and hasattr(quant, "scalar"):
                quant_s = f"scalar {quant.scalar.type.value}, always_ram={quant.scalar.always_ram}"
            per_agent = [(a.agent.name, await self.index.count(name, a.agent.id)) for a in agents]
            out.append(
                QdrantCollectionStats(
                    name=name,
                    status=info.status.value,
                    points=info.points_count or 0,
                    indexed_vectors=info.indexed_vectors_count or 0,
                    segments=info.segments_count,
                    vectors=vectors,
                    sparse_vectors=sparse,
                    hnsw=f"m={h.m}, payload_m={h.payload_m}, ef_construct={h.ef_construct}",
                    quantization=quant_s,
                    payload_indexes=[
                        f"{k}: {v.data_type.value} ({v.points} точек)"
                        for k, v in info.payload_schema.items()
                    ],
                    per_agent=[p for p in per_agent if p[1]],
                )
            )
        version = (await self._probe_http.get(f"{self.settings.qdrant_url}/")).json().get("version")
        return QdrantStats(version=version, collections=out)

    async def queue_depths(self) -> dict[str, int]:
        """Сообщений в очередях приложения (ready + unacked) для celery_queue_depth.

        Недоступный RabbitMQ — пустой словарь и warning: /metrics не должен отвечать 500.
        """
        try:
            resp = await self.rabbit_http.get("/api/queues")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            log.warning("metrics.queue_depth_failed", error=type(e).__name__)
            return {}
        return {
            q["name"]: q.get("messages_ready", 0) + q.get("messages_unacknowledged", 0)
            for q in resp.json()
            if not q["name"].startswith(("celery", "celeryev")) and ".pidbox" not in q["name"]
        }

    async def _celery(self) -> CeleryStats:
        queues_resp, overview_resp, workers = await asyncio.gather(
            self.rabbit_http.get("/api/queues"),
            self.rabbit_http.get("/api/overview"),
            self._workers(),
        )
        queues_resp.raise_for_status()
        queues = []
        for q in queues_resp.json():
            name = q["name"]
            # служебные очереди Celery (pidbox, celeryev) — шум для учебной схемы
            if name.startswith(("celery", "celeryev")) or ".pidbox" in name:
                continue
            stats = q.get("message_stats") or {}
            queues.append(
                QueueStats(
                    name=name,
                    ready=q.get("messages_ready", 0),
                    unacked=q.get("messages_unacknowledged", 0),
                    consumers=q.get("consumers", 0),
                    publish_rate=(stats.get("publish_details") or {}).get("rate", 0.0),
                    deliver_rate=(stats.get("deliver_get_details") or {}).get("rate", 0.0),
                    is_dlq=name.endswith(".dlq"),
                )
            )
        queues.sort(key=lambda q: (q.is_dlq, q.name))
        overview = overview_resp.json() if overview_resp.is_success else {}
        return CeleryStats(
            queues=queues,
            workers=workers,
            rabbit_version=overview.get("rabbitmq_version"),
            connections=(overview.get("object_totals") or {}).get("connections", 0),
        )

    async def _workers(self) -> dict[str, list[WorkerTask]] | None:
        """Что делают воркеры; None — не ответили (занят брокер, воркер перезапускается)."""
        try:
            # inspect() — синхронный broadcast через брокер: уводим в поток
            active = await asyncio.wait_for(
                anyio.to_thread.run_sync(self.inspect_workers), _PROBE_TIMEOUT_S
            )
        except Exception as e:  # сбой inspect не должен прятать очереди из management API
            log.info("system.inspect_failed", error=type(e).__name__)
            return None
        now = time.time()
        return {
            name: [
                WorkerTask(
                    name=t.get("name", "?"),
                    id=t.get("id", "?"),
                    runtime_s=round(now - t["time_start"], 1) if t.get("time_start") else None,
                    args=str(t.get("kwargs") or t.get("args") or "")[:200],
                )
                for t in tasks
            ]
            for name, tasks in active.items()
        }

    async def _redis(self) -> RedisStats:
        info = await self.redis.info()
        keyspace = {
            k: f"{v.get('keys', 0)} ключей, с TTL {v.get('expires', 0)}"
            for k, v in info.items()
            if k.startswith("db") and isinstance(v, dict)
        }
        sample: list[RedisKey] = []
        async for key in self.redis.scan_iter(count=200):
            if len(sample) >= _REDIS_SAMPLE:
                break
            sample.append(await self._describe_key(str(key)))
        sample.sort(key=lambda k: k.key)
        return RedisStats(
            version=info.get("redis_version"),
            used_memory=info.get("used_memory_human"),
            max_memory=info.get("maxmemory_human"),
            clients=int(info.get("connected_clients", 0)),
            ops_per_sec=int(info.get("instantaneous_ops_per_sec", 0)),
            keyspace=keyspace,
            sample_keys=sample,
        )

    async def _describe_key(self, key: str) -> RedisKey:
        kind = str(await self.redis.type(key))
        ttl = await self.redis.ttl(key)
        suffix = f" · TTL {ttl} с" if ttl > 0 else ""
        if kind == "hash":
            value = str(await self.redis.hgetall(key))[:120]
        elif kind == "list":
            value = f"{await self.redis.llen(key)} элементов"
        elif kind == "string":
            value = str(await self.redis.get(key))[:80]
        else:
            value = kind
        return RedisKey(key=key, kind=kind, value=value + suffix)

    async def _postgres(self) -> PostgresStats:
        async with self.db.session() as s:
            repo = PgStatsRepository(s)
            version = await repo.version()
            size = await repo.db_size()
            tables = await repo.tables()
        return PostgresStats(version=version, db_size=size, tables=tables)

    async def _ollama(self) -> OllamaStats:
        tags, ps = await asyncio.gather(
            self.ollama_http.get("/api/tags", timeout=2), self.ollama_http.get("/api/ps", timeout=2)
        )
        tags.raise_for_status()
        loaded = {m["name"] for m in ps.json().get("models", [])} if ps.is_success else set()
        models = []
        for m in tags.json().get("models", []):
            details = m.get("details") or {}
            models.append(
                OllamaModel(
                    name=m["name"],
                    family=details.get("family"),
                    parameters=details.get("parameter_size"),
                    quantization=details.get("quantization_level"),
                    size=_human_bytes(float(m.get("size", 0))),
                    loaded=m["name"] in loaded,
                )
            )
        return OllamaStats(base_url=self.settings.ollama_base_url, models=models)

    async def _admin_uis(self) -> list[AdminUi]:
        async def probe(spec: AdminUiSpec) -> AdminUi:
            try:
                await self._probe_http.get(spec.probe_url)
                up = True  # любой HTTP-ответ (даже 401) = сервис запущен
            except httpx.HTTPError:
                up = False
            return AdminUi(
                key=spec.key,
                title=spec.title,
                port=spec.port,
                path=spec.path,
                about=spec.about,
                up=up,
            )

        return list(await asyncio.gather(*(probe(s) for s in ADMIN_UIS)))

    async def vector_map(self, owner_id: UUID, agent_id: UUID) -> VectorMap | None:
        """2D-карта всех чанков агента (PCA) или None, если она ещё строится.

        Сборка — ~40 МБ JSON с векторами из Qdrant и PCA: в web она на десятки секунд занимала
        CPU и GIL процесса. Поэтому карту строит воркер (задача maintenance), web отдаёт готовую
        из Redis — общую для всех процессов и действительную до смены corpus_version агента.
        """
        async with self.db.session() as s:
            agent = await AgentRepository(s).get(owner_id, agent_id)
            target = await self._map_target(s, agent)
        if target is None:
            raise NotFoundError("agent")
        if (vmap := await self._cached_map(target)) is not None:
            return vmap
        # SET NX: из многих запросов задачу ставит один; TTL — на случай упавшего воркера
        lock = await self.redis.set(f"{target.redis_key}:building", "1", nx=True, ex=_MAP_LOCK_S)
        if lock:
            self.publisher.publish_build_vector_map(BuildVectorMapTask(agent_id=agent_id))
        return None

    async def build_vector_map(self, task: BuildVectorMapTask) -> None:
        """Воркер: собрать карту агента и положить в Redis. Владелец проверен при постановке."""
        async with self.db.session() as s:
            agent = await AgentRepository(s).get_unscoped(task.agent_id)
            target = await self._map_target(s, agent)
        if target is None:  # агента удалили, пока задача ждала в очереди
            return
        if await self._cached_map(target) is None:
            vmap = await self._build_map(target)
            await self.redis.set(target.redis_key, vmap.model_dump_json(), ex=_MAP_CACHE_TTL_S)
            log.info("system.vector_map_built", agent_id=str(task.agent_id), points=vmap.total)
        await self.redis.delete(f"{target.redis_key}:building")

    async def _map_target(self, s: AsyncSession, agent: AgentOut | None) -> _MapTarget | None:
        if agent is None or agent.active_index_id is None:
            return None
        index = await AgentRepository(s).get_index(agent.id, agent.active_index_id)
        if index is None:
            return None
        total = await self.index.count(index.collection, agent.id)
        return _MapTarget(agent.id, agent.corpus_version, total, index.collection, index.dim)

    async def _cached_map(self, target: _MapTarget) -> VectorMap | None:
        if (vmap := self._maps.get(target.agent_id)) and vmap[0] == target.redis_key:
            return vmap[1]
        raw = await self.redis.get(target.redis_key)
        if raw is None:
            return None
        parsed = VectorMap.model_validate_json(raw)
        # в памяти — только последняя версия карты агента
        self._maps[target.agent_id] = (target.redis_key, parsed)
        return parsed

    async def _build_map(self, target: _MapTarget) -> VectorMap:
        rows = await self.index.scroll_vectors(target.collection, target.agent_id, _MAP_LIMIT)
        rows.sort(key=lambda r: (r[2].document_id, r[2].ord))
        docs: dict[str, VectorMapDoc] = {}
        for _, _, p in rows:
            d = docs.setdefault(
                p.document_id,
                VectorMapDoc(id=p.document_id, title=p.book_title or p.document_id[:8], points=0),
            )
            d.points += 1
        doc_idx = {d: i for i, d in enumerate(docs)}
        matrix = np.array([r[1] for r in rows], dtype=np.float32).reshape(len(rows), target.dim)
        coords, mean, comps, explained = await anyio.to_thread.run_sync(pca_2d, matrix)
        return VectorMap(
            agent_id=str(target.agent_id),
            collection=target.collection,
            dim=target.dim,
            total=target.total,
            points=[
                VectorPoint(
                    id=pid,
                    x=round(float(xy[0]), 4),
                    y=round(float(xy[1]), 4),
                    doc=doc_idx[p.document_id],
                    ord=p.ord,
                    chapter=p.chapter_title,
                )
                for (pid, _, p), xy in zip(rows, coords, strict=True)
            ],
            docs=list(docs.values()),
            mean=[round(float(v), 5) for v in mean],
            components=[[round(float(v), 5) for v in c] for c in comps],
            explained=[round(e, 4) for e in explained],
        )

    async def point(self, owner_id: UUID, agent_id: UUID, point_id: UUID) -> PointDetail:
        """Чанк с полным вектором — для «рентгена» точки на карте."""
        async with self.db.session() as s:
            repo = AgentRepository(s)
            agent = await repo.get(owner_id, agent_id)
            index = (
                await repo.get_index(agent_id, agent.active_index_id)
                if agent and agent.active_index_id
                else None
            )
        if agent is None or index is None:
            raise NotFoundError("agent")
        found = await self.index.get_point(index.collection, agent_id, point_id)
        if found is None:
            raise NotFoundError("point")
        vec, p = found
        return PointDetail(
            id=str(point_id),
            book_title=p.book_title,
            chapter_title=p.chapter_title,
            ord=p.ord,
            text=p.text,
            vector=[round(v, 5) for v in vec],
            norm=round(float(np.linalg.norm(vec)), 4),
        )
