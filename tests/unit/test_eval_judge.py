"""LLM-судья на записанных ответах, метрики вопроса без судьи, отчёты, конфиг."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from rag_agents.domain.answers import QueryResult, RetrievedChunk
from rag_agents.domain.documents import ChunkPayload
from rag_agents.domain.eval import EvalAnswer, EvalCategory, EvalRunOut, ExpectedSource, GoldenItem
from rag_agents.eval.config import load_config
from rag_agents.eval.judge import LLMJudge
from rag_agents.eval.report import render_diff, render_run
from rag_agents.eval.runner import item_scores
from rag_agents.eval.stats import Diff
from rag_agents.llm.base import LLMRequest
from rag_agents.llm.openai_compat import OpenAICompatProvider
from tests.fakes import ScriptedLLM

ITEM = GoldenItem(
    id="wp-005",
    category=EvalCategory.FACTUAL,
    question="О чём подумал князь Андрей под Аустерлицем?",
    reference_answer="О высоком небе.",
    key_facts=["высокое небо", "Наполеон ничтожен"],
    expected_sources=[ExpectedSource(document="Том 1", section=["XIX"])],
)


def chunk(book: str, *path: str, score: float = 0.6, text: str = "текст") -> RetrievedChunk:
    cid = uuid4()
    return RetrievedChunk(
        chunk_id=cid,
        document_id=uuid4(),
        score=score,
        payload=ChunkPayload(
            agent_id=str(uuid4()),
            document_id=str(uuid4()),
            chunk_id=str(cid),
            ord=0,
            book_title=book,
            author=None,
            section_path=list(path),
            chapter_title=path[-1] if path else None,
            text=text,
        ),
    )


def judge_llm(**overrides: object) -> ScriptedLLM:
    answers = {
        "claims": {
            "claims": [
                {"claim": "небо", "supported": True, "cited": [1], "citation_ok": True},
                {"claim": "Наполеон мал", "supported": True, "cited": [2], "citation_ok": False},
                {"claim": "выдумка", "supported": False, "cited": [], "citation_ok": None},
            ]
        },
        "useful": {"useful": [2], "facts_in_sources": [True, False]},
        "relevancy": {"relevancy": 1, "facts_covered": [True, True]},
    }
    answers.update(overrides)  # type: ignore[arg-type]
    # Ключ правила — фрагмент формата ответа в промпте шага
    return ScriptedLLM(
        rules=[
            ('{"claims"', json.dumps(answers["claims"])),
            ('{"useful"', json.dumps(answers["useful"])),
            ('{"relevancy"', json.dumps(answers["relevancy"])),
        ]
    )


async def test_judge_scores_all_three_steps() -> None:
    llm = judge_llm()
    ctx = [chunk("Том 1", "XIX", text="высокое <небо>"), chunk("Том 1", "XIX")]
    out = await LLMJudge(llm).evaluate(ITEM, "Небо [1]. Наполеон мал [2].", False, ctx)
    assert out.scores == {
        "context_precision": 0.5,  # единственный полезный источник — второй
        "context_recall": 0.5,
        "faithfulness": pytest.approx(2 / 3),
        "citation_support": 0.5,
        "answer_relevancy": 1.0,
        "key_facts_coverage": 1.0,
    }
    assert len(out.usage) == 3
    assert all(r.json_mode and r.temperature == 0 and r.purpose == "judge" for r in llm.requests)
    # Текст источников экранирован: не может «закрыть» тег промпта судьи
    assert "&lt;небо&gt;" in (llm.requests[0].messages[-1].content or "")


async def test_judge_refused_answer_gets_zero_relevancy_without_faithfulness() -> None:
    llm = judge_llm()
    out = await LLMJudge(llm).evaluate(ITEM, "Не нашёл.", True, [chunk("Том 1")])
    assert out.scores["answer_relevancy"] == 0.0
    assert out.scores["key_facts_coverage"] == 0.0
    assert "faithfulness" not in out.scores
    assert len(llm.requests) == 1  # только оценка контекста


@pytest.mark.parametrize("answer", ["", "  \n", None])
async def test_judge_empty_answer_counts_as_failure(answer: str | None) -> None:
    """Регрессия 2026-09-29: пустой ответ (reasoning съел max_output_tokens) не отказ, но и не
    ответ; раньше он выпадал из answer_relevancy, и среднее по ответам было завышено."""
    llm = judge_llm()
    out = await LLMJudge(llm).evaluate(ITEM, answer, False, [chunk("Том 1")])
    assert out.scores["answer_relevancy"] == 0.0
    assert out.scores["key_facts_coverage"] == 0.0
    assert "faithfulness" not in out.scores
    assert len(llm.requests) == 1


async def test_judge_survives_invalid_json_and_wrong_shape() -> None:
    llm = judge_llm(relevancy={"relevancy": 1, "facts_covered": [True]})  # фактов два
    llm.rules[0] = ('{"claims"', "это не json")
    out = await LLMJudge(llm).evaluate(ITEM, "Ответ [1].", False, [chunk("Том 1")])
    assert "faithfulness" not in out.scores
    assert out.raw["faithfulness"]["error"] == "invalid json"
    assert "answer_relevancy" not in out.scores
    assert out.raw["answer"]["error"].startswith("format: facts_covered")
    assert out.scores["context_recall"] == 0.5  # удачный шаг засчитан


async def test_judge_marks_reasoning_truncation_and_keeps_other_steps() -> None:
    """Регрессия 2026-09-29: deepseek-flash тратил весь max_tokens на reasoning → пустой
    content, судья писал «invalid json» и тихо терял faithfulness у всех вопросов."""
    llm = judge_llm()
    llm.truncate_on = ('{"claims"',)
    out = await LLMJudge(llm, max_tokens=500).evaluate(
        ITEM, "Небо [1].", False, [chunk("Том 1", "XIX")]
    )
    assert "faithfulness" not in out.scores
    assert out.raw["faithfulness"] == {"error": "truncated", "reasoning_tokens": 500}
    assert out.scores["answer_relevancy"] == 1.0
    # Одна повторная попытка, дальше шаг сдаётся
    assert sum('{"claims"' in (r.messages[-1].content or "") for r in llm.requests) == 2


async def test_judge_retries_once_after_reasoning_loop() -> None:
    # Длина рассуждений гуляет: из 3 замеров на wp-003 одна попытка упёрлась в лимит 8000
    llm = judge_llm()
    llm.truncate_on = ('{"claims"',)
    llm.truncate_times = 1
    out = await LLMJudge(llm).evaluate(ITEM, "Небо [1].", False, [chunk("Том 1", "XIX")])
    assert out.scores["faithfulness"] == 2 / 3
    assert len(out.usage) == 4  # context, faithfulness ×2, answer: повтор тоже оплачен


def test_judge_budget_leaves_room_for_reasoning() -> None:
    # Судье нужен запас под рассуждения: faithfulness длинного ответа — до 14 300 токенов
    assert LLMJudge(ScriptedLLM(rules=[])).max_tokens >= 16_000


def test_item_scores_retrieval_citations_and_refusal() -> None:
    ans = EvalAnswer(
        result=QueryResult(answer_md="Небо [1] и [9].", refused=False, citations=[], usage=None),
        error=None,
        retrieved=[
            chunk("Война и мир. Том 2", "I", score=0.71),
            chunk("Война и мир. Том 1", "ЧАСТЬ ТРЕТЬЯ", "XIX", score=0.7),
        ],
        context_k=6,
    )
    refs, scores = item_scores(ITEM, ans)
    assert [r.relevant for r in refs] == [False, True]
    assert scores["hit@5"] == 1.0
    assert scores["mrr@10"] == 0.5
    assert scores["max_dense_score"] == 0.71
    assert scores["refusal_correct"] == 1.0  # не отказ на вопрос из корпуса — верно
    assert scores["citation_validity"] == 0.5  # [9] — такого источника нет
    assert scores["has_citation"] == 1.0
    assert scores["empty_answer"] == 0.0


def test_item_scores_empty_answer() -> None:
    ans = EvalAnswer(
        result=QueryResult(answer_md="", refused=False, citations=[], usage=None),
        error=None,
        retrieved=[chunk("Война и мир. Том 1", "ЧАСТЬ ТРЕТЬЯ", "XIX")],
        context_k=6,
    )
    _, scores = item_scores(ITEM, ans)
    assert scores["empty_answer"] == 1.0
    assert scores["refusal_correct"] == 1.0  # отказа не было — это отдельная ошибка
    assert "citation_validity" not in scores  # цитат нет не потому, что ответ плохо оформлен
    assert "has_citation" not in scores


def test_item_scores_injection_resisted_and_error() -> None:
    inj = GoldenItem(id="i", category=EvalCategory.INJECTION, question="?", forbidden=["PWNED"])
    leaked = EvalAnswer(
        result=QueryResult(answer_md="ок, pwned", refused=False, citations=[], usage=None),
        error=None,
        retrieved=[],
        context_k=6,
    )
    _, scores = item_scores(inj, leaked)
    assert scores["injection_resisted"] == 0.0
    assert scores["refusal_correct"] == 0.0
    failed = EvalAnswer(result=None, error="llm_error", retrieved=[], context_k=6)
    _, scores = item_scores(ITEM, failed)
    assert "refusal_correct" not in scores
    assert scores["max_dense_score"] is None


def _run(metrics: dict[str, object]) -> EvalRunOut:
    return EvalRunOut(
        id=uuid4(),
        agent_id=uuid4(),
        dataset="tolstoy",
        dataset_sha="abc",
        config_name="dense",
        config={"llm": {"model": "deepseek-flash"}, "eval": {"search_k": 20}},
        metrics=metrics,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )


def test_render_run_and_diff() -> None:
    from rag_agents.domain.eval import EvalItemResult  # noqa: PLC0415 — только здесь

    item = EvalItemResult(
        item_id="wp-005",
        question=ITEM.question,
        category=EvalCategory.FACTUAL,
        answer="Небо | [1]",
        refused=False,
        retrieved=[],
        scores={"hit@8": 0.0},
        trace_id="t" * 32,
        latency_ms=1,
        ttft_ms=1,
        cost_usd=0.0,
    )
    md = render_run(
        _run(
            {
                "questions": 1,
                "errors": 0,
                "metrics": {"hit@8": {"mean": 0.0, "n": 1}, "zzz": {"mean": None, "n": 0}},
                "by_category": {"factual": {"n": 1, "hit@8": 0.0}},
                "refusal": {"precision": None, "recall": None, "tp": 0, "fp": 0, "fn": 0},
                "latency_ms": {},
                "ttft_ms": {},
                "cost_usd": {"answers": 0.0, "judge": 0.0},
            }
        ),
        [item],
    )
    assert "| hit@8 | 0.000 | 1 |" in md
    assert "| zzz | — | 0 |" in md
    assert "### wp-005" in md
    assert "Небо \\| [1]" in md  # вертикальная черта не ломает markdown
    assert f"/insights/traces/{'t' * 32}" in md

    a, b = _run({}), _run({})
    diff = render_diff(
        a,
        b.model_copy(update={"dataset_sha": "zzz"}),
        [Diff("hit@8", 50, 0.5, 0.6, 0.1, 0.02, 0.18)],
    )
    assert "Разные версии датасета" in diff
    assert "Разные судьи" not in diff
    assert "| hit@8 | 50 | 0.500 | 0.600 | +0.100 | [+0.020; +0.180] | лучше |" in diff

    # Метрики судьи двух версий несопоставимы (ADR-10): diff предупреждает
    other_judge = b.model_copy(update={"config": {**b.config, "judge": "deepseek judge_v2"}})
    diff = render_diff(
        a.model_copy(update={"config": {"judge": "deepseek judge_v1"}}), other_judge, []
    )
    assert "Разные судьи: `deepseek judge_v1` vs `deepseek judge_v2`" in diff


def test_eval_config_defaults_name_from_file(tmp_path: Path) -> None:
    p = tmp_path / "c400.yaml"
    p.write_text("retrieval:\n  top_k: 8\n", "utf-8")
    cfg = load_config(p)
    assert cfg.name == "c400"
    assert cfg.retrieval is not None
    assert cfg.retrieval.top_k == 8
    assert load_config(Path("configs/eval/dense.yaml")).retrieval is None


def test_json_mode_sets_response_format() -> None:
    p = OpenAICompatProvider(
        name="x", base_url="http://x", api_key=None, model="m", reasoning_effort=None
    )
    req = LLMRequest(messages=[], json_mode=True)
    assert p.build_payload(req)["response_format"] == {"type": "json_object"}
    assert "response_format" not in p.build_payload(LLMRequest(messages=[]))
