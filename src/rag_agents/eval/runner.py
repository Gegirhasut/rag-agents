"""Прогон golden-датасета через прод-конвейер (QueryService.evaluate) и подсчёт метрик."""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog

from rag_agents.core.observability import Tracer
from rag_agents.domain.agents import AgentOut
from rag_agents.domain.eval import (
    EvalAnswer,
    EvalCategory,
    EvalItemResult,
    EvalRunOut,
    GoldenItem,
    RetrievedRef,
)
from rag_agents.domain.system import Node
from rag_agents.eval.config import EvalConfig
from rag_agents.eval.judge import JUDGE_VERSION, LLMJudge
from rag_agents.eval.metrics import aggregate, mark_relevant, norm, retrieval_scores
from rag_agents.llm.prices import PriceTable
from rag_agents.rag.prompting.citations import check_citations
from rag_agents.services.eval import EvalService
from rag_agents.services.query import QueryService
from rag_agents.services.trace import TraceBus

log = structlog.get_logger()

# Эти метрики уходят score-ами на трейс вопроса: в Langfuse видно, почему ответ плохой
TRACE_SCORES = (
    "hit@8",
    "mrr@10",
    "faithfulness",
    "answer_relevancy",
    "key_facts_coverage",
    "context_precision",
    "context_recall",
    "refusal_correct",
)
_ATTEMPTS = 2


@dataclass
class _Scored:
    item: EvalItemResult
    judge_cost: float


def item_scores(
    item: GoldenItem, ans: EvalAnswer
) -> tuple[list[RetrievedRef], dict[str, float | None]]:
    """Метрики вопроса без LLM-судьи: retrieval, отказ, цитаты, инъекция."""
    refs = mark_relevant(
        [
            RetrievedRef(
                chunk_id=c.chunk_id,
                book_title=c.payload.book_title,
                section_path=c.payload.section_path,
                score=c.score,
                relevant=False,
            )
            for c in ans.retrieved
        ],
        item.expected_sources,
    )
    scores = retrieval_scores(refs, item.expected_sources)
    # Сырьё для калибровки порога отказа (итерация 5): лучший dense-score вопроса
    scores["max_dense_score"] = refs[0].score if refs else None
    result = ans.result
    if result is None:
        return refs, scores
    scores["refusal_correct"] = float(result.refused == item.category.expects_refusal)
    if not result.refused:
        # Пустой ответ без отказа: у reasoning-модели весь max_output_tokens ушёл на рассуждения
        empty = not result.answer_md.strip()
        scores["empty_answer"] = float(empty)
        if not empty:
            check = check_citations(result.answer_md, min(len(ans.retrieved), ans.context_k))
            scores["citation_validity"] = check.validity
            scores["has_citation"] = float(check.valid > 0)
    if item.category == EvalCategory.INJECTION:
        answer = norm(result.answer_md)
        scores["injection_resisted"] = float(not any(norm(f) in answer for f in item.forbidden))
    return refs, scores


class EvalRunner:
    def __init__(
        self,
        query: QueryService,
        store: EvalService,
        tracer: Tracer,
        prices: PriceTable,
        judge: LLMJudge | None,
        trace: TraceBus | None = None,
    ) -> None:
        self.trace = trace
        self.query = query
        self.store = store
        self.tracer = tracer
        self.prices = prices
        self.judge = judge

    async def run(
        self,
        agent: AgentOut,
        items: list[GoldenItem],
        *,
        dataset: str,
        dataset_sha: str,
        config: EvalConfig,
        snapshot: dict[str, Any],
        on_item: Callable[[EvalItemResult], None] | None = None,
    ) -> tuple[EvalRunOut, list[EvalItemResult]]:
        """Все вопросы (параллельно по config.concurrency) → eval_items → сводка в eval_runs."""
        run = await self.store.start_run(
            agent.id,
            dataset=dataset,
            dataset_sha=dataset_sha,
            config_name=config.name,
            config=snapshot,
        )
        log.info("eval.started", run_id=str(run.id), agent_id=str(agent.id), items=len(items))
        await self._emit(
            "eval.started",
            "postgres",
            f"eval_runs: прогон «{config.name}», датасет {dataset}, вопросов: {len(items)}",
            agent,
        )
        sem = asyncio.Semaphore(config.concurrency)

        async def one(item: GoldenItem) -> _Scored:
            async with sem:
                scored = await self._item(run.id, agent, item, config)
            await self.store.save_item(run.id, scored.item)
            if on_item is not None:
                on_item(scored.item)
            return scored

        scored = await asyncio.gather(*(one(it) for it in items))
        results = [s.item for s in scored]
        summary = aggregate(results)
        summary["cost_usd"]["judge"] = sum(s.judge_cost for s in scored)
        run = await self.store.finish_run(agent.id, run.id, summary)
        overall = summary["metrics"]
        headline = ", ".join(
            f"{k} {overall[k]['mean']:.2f}"
            for k in ("hit@8", "faithfulness", "refusal_correct")
            if k in overall and overall[k]["mean"] is not None
        )
        await self._emit(
            "eval.finished", "postgres", f"Сводка прогона → eval_runs.metrics: {headline}", agent
        )
        await asyncio.to_thread(self.tracer.flush)
        log.info("eval.finished", run_id=str(run.id))
        return run, results

    async def _answer(
        self, run_id: UUID, agent: AgentOut, item: GoldenItem, config: EvalConfig
    ) -> EvalAnswer:
        last: EvalAnswer | None = None
        for _ in range(_ATTEMPTS):
            try:
                last = await self.query.evaluate(
                    agent.owner_id,
                    agent.id,
                    item.question,
                    run_id=run_id,
                    item_id=item.id,
                    retrieval=config.retrieval,
                    search_k=config.search_k,
                )
            except Exception as e:  # вопрос с ошибкой не должен валить весь прогон
                log.warning("eval.item_failed", item_id=item.id, error=repr(e)[:300])
                last = EvalAnswer(result=None, error=type(e).__name__, retrieved=[], context_k=0)
            if last.error is None:
                return last
        assert last is not None  # noqa: S101 — цикл выполняется хотя бы раз
        return last

    async def _item(
        self, run_id: UUID, agent: AgentOut, item: GoldenItem, config: EvalConfig
    ) -> _Scored:
        t0 = time.monotonic()
        await self._emit(
            "eval.question",
            None,
            f"Вопрос {item.id} ({item.category.value}) → тот же QueryService, что у пользователя",
            agent,
        )
        ans = await self._answer(run_id, agent, item, config)
        refs, scores = item_scores(item, ans)
        result = ans.result
        judge_raw: dict[str, Any] | None = None
        judge_cost = 0.0
        if (
            self.judge is not None
            and config.judge
            and result is not None
            and not item.category.expects_refusal
        ):
            await self._emit(
                "eval.judge",
                "llm",
                f"Судья {JUDGE_VERSION} по {item.id}: faithfulness · context · answer (3 вызова)",
                agent,
            )
            verdict = await self.judge.evaluate(
                item, result.answer_md, result.refused, ans.retrieved[: ans.context_k]
            )
            scores.update(verdict.scores)
            for judge_usage in verdict.usage:
                cost = self.prices.cost(
                    self.judge.llm.name, self.judge.llm.model, judge_usage, datetime.now(UTC)
                )
                judge_cost += cost.total if cost else 0.0
            judge_raw = {"model": self.judge.model, "version": JUDGE_VERSION, **verdict.raw}
        usage = result.usage if result else None
        out = EvalItemResult(
            item_id=item.id,
            question=item.question,
            category=item.category,
            answer=result.answer_md if result else None,
            error=ans.error,
            refused=result.refused if result else None,
            retrieved=refs,
            scores=scores,
            judge=judge_raw,
            trace_id=ans.trace_id,
            latency_ms=usage.t_total_ms if usage else int((time.monotonic() - t0) * 1000),
            ttft_ms=usage.t_first_token_ms if usage else None,
            cost_usd=usage.cost_usd if usage else None,
        )
        self._score_trace(out, run_id)
        shown = ("hit@8", "faithfulness", "answer_relevancy", "refusal_correct")
        await self._emit(
            "eval.scored",
            "postgres",
            f"eval_items: {item.id} — "
            + (
                ", ".join(f"{k}={scores[k]:.2f}" for k in shown if scores.get(k) is not None)
                or "без метрик"
            ),
            agent,
        )
        return _Scored(out, judge_cost)

    async def _emit(self, kind: str, dst: Node | None, label: str, agent: AgentOut) -> None:
        """Шаг прогона на живой схеме /system (узел eval); без шины событий — ничего."""
        if self.trace is not None:
            await self.trace.emit(kind, "eval", dst, label, agent_id=agent.id)

    def _score_trace(self, item: EvalItemResult, run_id: UUID) -> None:
        if item.trace_id is None:
            return
        for key in TRACE_SCORES:
            value = item.scores.get(key)
            if value is not None:
                self.tracer.score(
                    trace_id=item.trace_id,
                    name=f"eval_{key.replace('@', '_at_')}",
                    value=value,
                    score_id=f"eval-{run_id}-{item.item_id}-{key}",
                )
