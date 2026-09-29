"""Метрики eval, bootstrap, датасет, цитаты — чистые функции."""

import json
from pathlib import Path
from uuid import uuid4

import pytest

from rag_agents.domain.eval import (
    EvalCategory,
    EvalItemResult,
    ExpectedSource,
    GoldenItem,
    RetrievedRef,
)
from rag_agents.eval.dataset import DatasetError, category_shares, load_dataset
from rag_agents.eval.metrics import (
    aggregate,
    average_precision,
    mark_relevant,
    matches,
    mrr_at_k,
    norm,
    percentile,
    refusal_confusion,
    retrieval_scores,
    worst,
)
from rag_agents.eval.stats import compare, paired_bootstrap
from rag_agents.rag.prompting.citations import check_citations, cited_numbers, is_grounded

DATASET = Path("eval/datasets/tolstoy.jsonl")


def ref(book: str, *path: str, score: float = 0.5, relevant: bool = False) -> RetrievedRef:
    return RetrievedRef(
        chunk_id=uuid4(), book_title=book, section_path=list(path), score=score, relevant=relevant
    )


def result(
    item_id: str,
    category: EvalCategory = EvalCategory.FACTUAL,
    *,
    refused: bool | None = False,
    **scores: float | None,
) -> EvalItemResult:
    return EvalItemResult(
        item_id=item_id,
        question="?",
        category=category,
        answer="ответ",
        refused=refused,
        retrieved=[],
        scores=scores,
        latency_ms=100,
        ttft_ms=50,
        cost_usd=0.001,
    )


# --- сопоставление источников ---


def test_norm_folds_latin_homoglyphs_case_and_yo() -> None:
    # В fb2 «Войны и мира» заголовок набран с латинскими B и T
    assert norm("ЧАСТЬ BTОРАЯ") == norm("Часть вторая")
    assert norm("  Ёлка  и  ель ") == "елка и ель"
    assert norm("Х") == norm("X")  # глава X кириллицей и латиницей


def test_source_matches_document_substring_and_section_subsequence() -> None:
    chunk = ref("Война и мир. Том 1", "ЧАСТЬ BTОРАЯ", "XV")
    assert matches(chunk, ExpectedSource(document="Том 1"))
    assert matches(chunk, ExpectedSource(document="война и мир. том 1", section=["XV"]))
    assert matches(chunk, ExpectedSource(document="Том 1", section=["Часть вторая", "XV"]))
    assert not matches(chunk, ExpectedSource(document="Том 2", section=["XV"]))
    assert not matches(chunk, ExpectedSource(document="Том 1", section=["XVI"]))
    # Порядок важен: подпоследовательность, а не множество
    assert not matches(chunk, ExpectedSource(document="Том 1", section=["XV", "Часть вторая"]))


def test_retrieval_scores_hit_recall_mrr() -> None:
    expected = [
        ExpectedSource(document="Том 1", section=["XIX"]),
        ExpectedSource(document="Исповедь", section=["IV"]),
    ]
    refs = mark_relevant(
        [
            ref("Война и мир. Том 2", "I"),
            ref("Война и мир. Том 1", "ЧАСТЬ ТРЕТЬЯ", "XIX"),
            *[ref("Война и мир. Том 3", str(i)) for i in range(10)],
            ref("Исповедь", "ИСПОВЕДЬ", "IV"),
        ],
        expected,
    )
    assert [r.relevant for r in refs[:2]] == [False, True]
    scores = retrieval_scores(refs, expected)
    assert scores["hit@5"] == 1.0
    assert scores["recall@5"] == 0.5  # второй источник только на 13-м месте
    assert scores["recall@20"] == 1.0
    assert scores["mrr@10"] == 0.5
    assert retrieval_scores(refs, []) == {}  # вопросы без источников (отказные) — без метрик


def test_mrr_zero_when_nothing_relevant_in_top_k() -> None:
    assert mrr_at_k([ref("x")] * 3, 10) == 0.0
    assert mrr_at_k([*[ref("x")] * 10, ref("x", relevant=True)], 10) == 0.0


def test_average_precision_rewards_relevant_on_top() -> None:
    assert average_precision([True, False, False]) == 1.0
    assert average_precision([False, False, True]) == pytest.approx(1 / 3)
    assert average_precision([True, False, True]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision([False, False]) == 0.0


def test_percentile_nearest_rank() -> None:
    assert percentile([], 50) is None
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile(list(range(1, 101)), 95) == 95


# --- сводка ---


def test_refusal_confusion_counts_by_category() -> None:
    items = [
        result("a", EvalCategory.OUT_OF_CORPUS, refused=True),  # верный отказ
        result("b", EvalCategory.INJECTION, refused=False),  # пропущенный
        result("c", EvalCategory.FACTUAL, refused=True),  # лишний
        result("d", EvalCategory.FACTUAL, refused=False),
        result("e", EvalCategory.FACTUAL, refused=None),  # ошибка — не считается
    ]
    conf = refusal_confusion(items)
    assert (conf["tp"], conf["fp"], conf["fn"]) == (1, 1, 1)
    assert conf["precision"] == 0.5
    assert conf["recall"] == 0.5


def test_aggregate_means_over_defined_values_only() -> None:
    items = [
        result("a", **{"hit@8": 1.0, "faithfulness": 0.5}),
        result("b", **{"hit@8": 0.0, "faithfulness": None}),
        result("c", EvalCategory.OUT_OF_CORPUS, refused=True, refusal_correct=1.0),
    ]
    agg = aggregate(items)
    assert agg["questions"] == 3
    assert agg["metrics"]["hit@8"] == {"mean": 0.5, "n": 2}
    assert agg["metrics"]["faithfulness"] == {"mean": 0.5, "n": 1}
    assert agg["by_category"]["factual"]["n"] == 2
    assert agg["by_category"]["out_of_corpus"]["refusal_correct"] == 1.0
    assert agg["latency_ms"] == {"p50": 100, "p95": 100}
    assert agg["cost_usd"]["answers"] == pytest.approx(0.003)


def test_worst_puts_errors_and_misses_first() -> None:
    good = result("good", **{"hit@8": 1.0, "key_facts_coverage": 1.0, "faithfulness": 1.0})
    bad = result("bad", **{"hit@8": 0.0, "key_facts_coverage": 0.0})
    missed_refusal = result("ooc", EvalCategory.OUT_OF_CORPUS, refusal_correct=0.0)
    err = result("err").model_copy(update={"error": "llm_error"})
    assert [it.item_id for it in worst([good, bad, missed_refusal, err], 3)] == [
        "err",
        "bad",
        "ooc",
    ]


# --- bootstrap ---


def test_paired_bootstrap_identical_runs_have_zero_ci() -> None:
    a = [0.0, 1.0, 1.0, 0.5] * 10
    assert paired_bootstrap(a, a) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_detects_consistent_improvement_and_noise() -> None:
    a = [0.2 + 0.01 * i for i in range(50)]
    diff, lo, hi = paired_bootstrap(a, [x + 0.1 for x in a])
    assert diff == pytest.approx(0.1)
    assert lo > 0
    noisy = [x + (0.1 if i % 2 else -0.1) for i, x in enumerate(a)]
    _, lo, hi = paired_bootstrap(a, noisy)
    assert lo < 0 < hi


def test_paired_bootstrap_is_reproducible_and_validates_input() -> None:
    a, b = [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]
    assert paired_bootstrap(a, b, seed=1) == paired_bootstrap(a, b, seed=1)
    with pytest.raises(ValueError, match="одинаковой длины"):
        paired_bootstrap([1.0], [1.0, 2.0])


def test_compare_pairs_by_item_id_and_skips_undefined() -> None:
    a = [result("q1", **{"hit@8": 0.0}), result("q2", **{"hit@8": 1.0}), result("q3")]
    b = [result("q2", **{"hit@8": 1.0}), result("q1", **{"hit@8": 1.0}), result("q4")]
    [d] = compare(a, b, ["hit@8", "faithfulness"])
    assert (d.metric, d.n, d.mean_a, d.mean_b) == ("hit@8", 2, 0.5, 1.0)
    assert d.verdict in {"лучше", "шум"}


def test_compare_lower_is_better_for_empty_answers() -> None:
    # Пустых ответов стало меньше во всех 20 вопросах — это улучшение, хотя B − A < 0
    a = [result(f"q{i}", empty_answer=1.0) for i in range(20)]
    b = [result(f"q{i}", empty_answer=0.0) for i in range(20)]
    [d] = compare(a, b, ["empty_answer"])
    assert d.diff == -1.0
    assert d.verdict == "лучше"


# --- цитаты ---


def test_citations_parse_lists_and_ignore_notes() -> None:
    assert cited_numbers("Так [1], и так [2, 3]; [прим. 4] и (1812)") == [1, 2, 3]
    check = check_citations("Факт [1] и [7] и [2,1]", n_sources=3)
    assert (check.markers, check.valid, check.used) == (4, 3, frozenset({1, 2}))
    assert check.validity == 0.75
    assert check_citations("без ссылок", 3).validity is None


def test_grounded_means_refusal_or_valid_citation() -> None:
    assert is_grounded("Не нашёл.", refused=True, n_sources=0)
    assert is_grounded("Да [2].", refused=False, n_sources=2)
    assert not is_grounded("Да [3].", refused=False, n_sources=2)
    assert not is_grounded("Да.", refused=False, n_sources=2)


# --- датасет ---


def _write(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), "utf-8")
    return p


FACT = {
    "id": "f1",
    "category": "factual",
    "question": "?",
    "reference_answer": "!",
    "expected_sources": [{"document": "Исповедь"}],
}


def test_dataset_validation(tmp_path: Path) -> None:
    items, sha = load_dataset(
        _write(tmp_path, [FACT, {"id": "o1", "category": "out_of_corpus", "question": "?"}])
    )
    assert [i.id for i in items] == ["f1", "o1"]
    assert len(sha) == 12
    with pytest.raises(DatasetError, match="повторяются"):
        load_dataset(_write(tmp_path, [FACT, FACT]))
    with pytest.raises(DatasetError, match="нет источников"):
        load_dataset(_write(tmp_path, [{**FACT, "category": "out_of_corpus"}]))
    with pytest.raises(DatasetError, match="нужны expected_sources"):
        load_dataset(_write(tmp_path, [{**FACT, "expected_sources": []}]))
    with pytest.raises(DatasetError, match="Extra inputs"):
        load_dataset(_write(tmp_path, [{**FACT, "chapter": "I"}]))


def test_golden_tolstoy_dataset_follows_architecture() -> None:
    """ARCHITECTURE §15.1: 40–60 вопросов, ≥ 20 % с ожидаемым отказом, все категории."""
    items, _ = load_dataset(DATASET)
    assert 40 <= len(items) <= 60
    shares = category_shares(items)
    assert all(shares[c] > 0 for c in EvalCategory)
    assert shares[EvalCategory.OUT_OF_CORPUS] >= 0.2
    for it in items:
        if not it.category.expects_refusal:
            assert it.key_facts, f"{it.id}: ключевые факты нужны судье"
            assert all(s.document for s in it.expected_sources)
        if it.category == EvalCategory.INJECTION:
            assert it.forbidden, f"{it.id}: маркер выполненной инъекции"


def test_golden_item_requires_consistency() -> None:
    with pytest.raises(ValueError, match="нет источников"):
        GoldenItem(
            id="x",
            category=EvalCategory.INJECTION,
            question="?",
            expected_sources=[ExpectedSource(document="a")],
        )
