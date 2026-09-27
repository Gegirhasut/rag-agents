import numpy as np

from rag_agents.domain.system import (
    CeleryStats,
    LlmInfo,
    OllamaStats,
    PostgresStats,
    QdrantStats,
    QueueStats,
    RedisStats,
    SystemSnapshot,
)
from rag_agents.services.system import pca_2d
from rag_agents.web.routes.system import _badges
from rag_agents.web.system_catalog import EDGES, NODES


def test_pca_keeps_dominant_direction() -> None:
    rng = np.random.default_rng(0)
    t = rng.normal(size=200)
    # точки вытянуты вдоль оси 3 из 8: первая компонента должна её найти
    x = rng.normal(scale=0.01, size=(200, 8))
    x[:, 3] += t
    coords, mean, comps, explained = pca_2d(x)
    assert coords.shape == (200, 2)
    assert mean.shape == (8,)
    assert abs(comps[0][3]) > 0.99
    assert explained[0] > 0.9
    # проекция в браузере: (v - mean) · comp — та же формула
    np.testing.assert_allclose((x[5] - mean) @ comps.T, coords[5], atol=1e-9)


def test_pca_degenerate_inputs() -> None:
    coords, _, comps, explained = pca_2d(np.zeros((1, 4)))
    assert coords.shape == (1, 2)
    assert comps.shape == (2, 4)
    assert explained == [0.0, 0.0]
    assert pca_2d(np.zeros((0, 4)))[0].shape == (0, 2)


def test_schema_edges_reference_known_nodes() -> None:
    ids = {n.id for n in NODES}
    assert {a for e in EDGES for a in e} <= ids


def _snapshot(celery: CeleryStats) -> SystemSnapshot:
    return SystemSnapshot(
        qdrant=QdrantStats(),
        celery=celery,
        redis=RedisStats(),
        postgres=PostgresStats(),
        ollama=OllamaStats(base_url="http://x"),
        llm=LlmInfo(base_url="u", model="m", reasoning_effort=None, key_configured=False),
        admin_uis=[],
    )


def test_badge_distinguishes_silent_inspect_from_no_workers() -> None:
    assert _badges(_snapshot(CeleryStats(workers=None)))["worker"] == "нет данных"
    assert _badges(_snapshot(CeleryStats(workers={"w@1": []})))["worker"] == "простаивает"


def test_badges_warn_when_no_workers() -> None:
    snap = SystemSnapshot(
        qdrant=QdrantStats(),
        celery=CeleryStats(
            workers={},
            queues=[
                QueueStats(
                    name="ingest.parse.dlq",
                    ready=2,
                    unacked=0,
                    consumers=0,
                    publish_rate=0,
                    deliver_rate=0,
                    is_dlq=True,
                )
            ],
        ),
        redis=RedisStats(),
        postgres=PostgresStats(),
        ollama=OllamaStats(base_url="http://x"),
        llm=LlmInfo(base_url="u", model="m", reasoning_effort=None, key_configured=False),
        admin_uis=[],
    )
    badges = _badges(snap)
    assert badges["worker"] == "воркеров нет!"
    assert "DLQ 2" in badges["rabbitmq"]
    assert badges["llm"] == "m · нет ключа"
    assert set(badges) == {n.id for n in NODES}
