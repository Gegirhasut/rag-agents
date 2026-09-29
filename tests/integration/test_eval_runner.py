"""Мини-eval (ARCHITECTURE §15.2, «регрессия»): фикстурный корпус → тот же QueryService →
метрики → eval_runs/eval_items → отчёт и diff. Без сети: эмбеддер по ключевым словам,
ответы и судья — записанные.

Ключевой эмбеддер делает retrieval осмысленным, поэтому тест ловит и регрессии самого
поиска (фильтр agent_id, top-k, сопоставление источников), а не только «прогон не упал».
"""

import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from rag_agents.container import Container, build_container
from rag_agents.core.config import Settings
from rag_agents.core.observability import NoopTracer
from rag_agents.domain.agents import AgentCreate, AgentOut, RetrievalSettings
from rag_agents.domain.auth import UserOut
from rag_agents.eval.config import EvalConfig
from rag_agents.eval.dataset import load_dataset
from rag_agents.eval.judge import LLMJudge
from rag_agents.eval.report import render_run
from rag_agents.eval.runner import EvalRunner
from rag_agents.eval.stats import compare
from rag_agents.llm.prices import PriceTable
from rag_agents.rag.chunking.structural import StructuralChunker
from rag_agents.rag.prompting.builder import REFUSAL_TEXT
from rag_agents.services.ingest import IngestService
from rag_agents.services.query import QueryService
from tests.fakes import (
    KeywordEmbedder,
    RecordingPublisher,
    RecordingTraceBus,
    RecordingTracer,
    ScriptedLLM,
    WordCounter,
)
from tests.integration.conftest import DATABASE_URL, PASSWORD, QDRANT_URL, REDIS_URL

pytestmark = pytest.mark.integration
DATASET = Path(__file__).parents[1] / "fixtures" / "eval_mini.jsonl"

WAR_AND_PEACE = """ГЛАВА I

Князь Андрей ехал через лес и увидел старый дуб. Дуб стоял без листьев, корявый дуб.

ГЛАВА II

Раненый князь Андрей увидел высокое небо. Небо было бесконечно, и небо было тихо.

ГЛАВА III

На бале Наташа ждала приглашения. Князь Андрей пригласил её на тур вальса, и бал закружился.
"""
CONFESSION = """ГЛАВА IV

Путник висит над колодцем, внизу дракон, а две мыши, чёрная и белая, грызут ветку.
Дракон — это смерть, мыши — дни и ночи.

ГЛАВА V

Вера есть знание смысла жизни. Вера есть сила жизни, и без веры жить нельзя.
"""

# Ответ системы: отказ на «борщ» и на инъекцию, на остальное — ответ со ссылкой
ANSWERS = ScriptedLLM(
    rules=[("борщ", REFUSAL_TEXT), ("PWNED", REFUSAL_TEXT)],
    default="Ответ по источнику [1].",
)


def judge_llm() -> ScriptedLLM:
    claims = {"claims": [{"claim": "x", "supported": True, "cited": [1], "citation_ok": True}]}
    return ScriptedLLM(
        rules=[
            ('{"claims"', json.dumps(claims)),
            # два ключевых факта только у m-05; судье передаётся их нумерованный список
            (
                ('{"useful"', "2. вера"),
                json.dumps({"useful": [1], "facts_in_sources": [True, False]}),
            ),
            ('{"useful"', json.dumps({"useful": [1], "facts_in_sources": [True]})),
            (
                ('{"relevancy"', "2. вера"),
                json.dumps({"relevancy": 0.5, "facts_covered": [True, False]}),
            ),
            ('{"relevancy"', json.dumps({"relevancy": 1, "facts_covered": [True]})),
        ]
    )


async def _bytes(data: bytes) -> AsyncIterator[bytes]:
    yield data


@pytest.fixture
async def mini(tmp_path: Path) -> AsyncIterator[tuple[Container, RecordingPublisher]]:
    settings = Settings(
        app_env="test",
        database_url=DATABASE_URL,
        redis_url=REDIS_URL,
        qdrant_url=QDRANT_URL,
        qdrant_collection_prefix=f"e{uuid.uuid4().hex[:8]}_",
        embedding_model=KeywordEmbedder.model,
        embedding_dim=KeywordEmbedder.dim,
        upload_dir=tmp_path,
        trace_enabled=False,
    )
    publisher = RecordingPublisher()
    c = build_container(settings, publisher)
    embedder = KeywordEmbedder()
    c.query = QueryService(
        c.db, embedder, c.query.index, ANSWERS, c.trace, settings, RecordingTracer(), PriceTable({})
    )
    c._ingest = IngestService(
        c.db,
        c.documents.storage,
        lambda: StructuralChunker(WordCounter(), target=40, max_tokens=60, min_tokens=5, overlap=0),
        embedder,
        c.query.index,
        c.documents.progress,
        c.trace,
        settings,
        NoopTracer(),
        publisher,
    )
    yield c, publisher
    await c.aclose()


async def _corpus(c: Container, publisher: RecordingPublisher) -> tuple[UserOut, AgentOut]:
    user = await c.auth.create_user(
        f"eval-{uuid.uuid4().hex[:6]}@t.local", PASSWORD, is_admin=False
    )
    agent = await c.agents.create(user.id, AgentCreate(name="Мини-Толстой"))
    for name, text in (
        ("Л. Н. Толстой - Война и мир (фрагмент).txt", WAR_AND_PEACE),
        ("Л. Н. Толстой - Исповедь.txt", CONFESSION),
    ):
        await c.documents.upload(user.id, agent.id, name, _bytes(text.encode()))
    while publisher.tasks or publisher.embeds:
        if publisher.tasks:
            await c.ingest.parse(publisher.tasks.pop(0), final_attempt=False)
        else:
            await c.ingest.embed_batch(publisher.embeds.pop(0), final_attempt=False)
    return user, agent


async def test_mini_eval_retrieval_metrics_and_persistence(
    mini: tuple[Container, RecordingPublisher], tmp_path: Path
) -> None:
    c, publisher = mini
    _, agent = await _corpus(c, publisher)
    golden, sha = load_dataset(DATASET)
    runner = EvalRunner(c.query, c.evals, NoopTracer(), PriceTable({}), LLMJudge(judge_llm()))
    config = EvalConfig(name="dense", retrieval=RetrievalSettings(top_k=2), search_k=5)
    seen: list[str] = []

    run, results = await runner.run(
        agent,
        golden,
        dataset="eval_mini",
        dataset_sha=sha,
        config=config,
        snapshot={"eval": config.model_dump(mode="json")},
        on_item=lambda it: seen.append(it.item_id),
    )
    by_id = {r.item_id: r for r in results}
    assert sorted(seen) == [g.id for g in golden]

    # Retrieval: ключевые слова вопроса находят нужную главу первой
    for item_id in ("m-01", "m-02", "m-03", "m-04"):
        assert by_id[item_id].scores["mrr@10"] == 1.0, item_id
        assert by_id[item_id].retrieved[0].relevant
    # multi_hop: оба источника в top-5 (recall@5), причём «небо» — первым
    assert by_id["m-05"].scores["recall@5"] == 1.0
    # Отказы: верные на вопросы вне корпуса и на инъекцию
    assert by_id["m-06"].refused is True
    assert by_id["m-06"].scores["refusal_correct"] == 1.0
    assert by_id["m-07"].scores["injection_resisted"] == 1.0
    assert "hit@5" not in by_id["m-06"].scores  # у отказных нет источников
    # Судья: m-05 покрыт наполовину, остальные — полностью
    assert by_id["m-05"].scores["key_facts_coverage"] == 0.5
    assert by_id["m-01"].scores["faithfulness"] == 1.0
    assert by_id["m-01"].judge is not None
    assert by_id["m-06"].judge is None  # на отказные вопросы судья не зовётся
    # Цитаты: [1] при двух источниках в промпте — валидна
    assert by_id["m-01"].scores["citation_validity"] == 1.0

    m = run.metrics
    assert m is not None
    assert run.finished_at is not None
    assert m["questions"] == len(golden)
    assert m["metrics"]["hit@5"]["mean"] == 1.0
    assert m["refusal"]["precision"] == 1.0
    assert m["refusal"]["recall"] == 1.0
    assert run.config["eval"]["search_k"] == 5

    # PG: прогон и вопросы читаются обратно (diff и отчёт идут отсюда)
    stored_run, stored = await c.evals.get_run(run.id)
    assert stored_run.dataset_sha == sha
    assert [s.item_id for s in stored] == sorted(g.id for g in golden)
    assert stored[0].retrieved[0].chunk_id == by_id["m-01"].retrieved[0].chunk_id
    assert [r.id for r in await c.evals.list_runs(agent.id)] == [run.id]

    md = render_run(stored_run, stored)
    assert "| hit@5 | 1.000 |" in md
    assert "## 10 худших примеров" in md

    # Ответы eval не попадают в чаты пользователя (и в /insights)
    assert await c.query.history(agent.owner_id, agent.id) == []

    # Второй прогон с тем же поведением: diff без значимых разниц
    run2, results2 = await runner.run(
        agent,
        golden,
        dataset="eval_mini",
        dataset_sha=sha,
        config=config,
        snapshot={},
    )
    diffs = {d.metric: d for d in compare(results, results2, ["hit@5", "mrr@10"])}
    assert diffs["hit@5"].diff == 0.0
    assert diffs["hit@5"].verdict == "шум"
    assert run2.id != run.id


async def test_eval_trace_is_separate_session_and_not_saved_to_chat(
    mini: tuple[Container, RecordingPublisher],
) -> None:
    c, publisher = mini
    tracer = RecordingTracer()
    c.query.tracer = tracer
    user, agent = await _corpus(c, publisher)
    run_id = uuid.uuid4()
    ans = await c.query.evaluate(
        user.id,
        agent.id,
        "Какой дуб?",
        run_id=run_id,
        item_id="m-01",
        retrieval=RetrievalSettings(top_k=2),
        search_k=4,
    )
    assert ans.error is None
    assert ans.result is not None
    assert len(ans.retrieved) == 4  # кандидаты для метрик
    assert ans.context_k == 2  # в промпт ушли top_k эксперимента, а не агента
    assert len(ans.result.citations) == 2
    [root] = tracer.roots()
    assert root.trace_id == tracer.trace_id_for(f"eval:{run_id}:m-01")
    assert root.trace_attrs["session_id"] == f"eval-{run_id}"
    assert root.trace_attrs["tags"] == ["Мини-Толстой", "eval"]
    assert ans.trace_id == root.trace_id


async def test_eval_run_animates_its_own_node_on_live_scheme(
    mini: tuple[Container, RecordingPublisher],
) -> None:
    """На /system прогон виден как узел eval: тот же конвейер, но без web и браузера."""
    c, publisher = mini
    _, agent = await _corpus(c, publisher)
    bus = RecordingTraceBus()
    c.query.trace = bus  # type: ignore[assignment]  # фейк шины без Redis
    golden, sha = load_dataset(DATASET)
    runner = EvalRunner(
        c.query,
        c.evals,
        NoopTracer(),
        PriceTable({}),
        LLMJudge(judge_llm()),
        bus,  # type: ignore[arg-type]
    )
    config = EvalConfig(name="dense", retrieval=RetrievalSettings(top_k=2), search_k=5)
    await runner.run(
        agent, golden[:2], dataset="eval_mini", dataset_sha=sha, config=config, snapshot={}
    )

    nodes = {n for _, src, dst in bus.events for n in (src, dst) if n}
    assert "web" not in nodes
    assert "browser" not in nodes
    edges = {(src, dst) for _, src, dst in bus.events}
    assert {("eval", "ollama"), ("ollama", "eval"), ("eval", "qdrant"), ("eval", "llm")} <= edges
    kinds = [k for k, _, _ in bus.events]
    assert kinds[0] == "eval.started"
    assert kinds[-1] == "eval.finished"
    assert kinds.count("eval.question") == kinds.count("eval.scored") == 2
    assert kinds.count("eval.judge") == 2
    assert "query.done" in kinds  # ответ готов — внутри узла eval, без записи в чат
    assert ("query.done", "eval", None) in bus.events
