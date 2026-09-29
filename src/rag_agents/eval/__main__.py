"""CLI eval: `python -m rag_agents.eval run|diff|list` (на стенде — make eval / eval-diff)."""

import asyncio
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

import structlog
import typer

from rag_agents.container import Container, build_container
from rag_agents.core.config import get_settings
from rag_agents.core.langfuse_api import LangfuseApiError, LangfuseDatasets
from rag_agents.core.logging import configure_logging
from rag_agents.domain.agents import AgentOut
from rag_agents.domain.eval import EvalItemResult, EvalRunOut, GoldenItem
from rag_agents.eval.config import EvalConfig, load_config
from rag_agents.eval.dataset import load_dataset
from rag_agents.eval.judge import JUDGE_VERSION, LLMJudge
from rag_agents.eval.report import ORDER, render_diff, render_run
from rag_agents.eval.runner import EvalRunner
from rag_agents.eval.stats import compare
from rag_agents.rag.prompting.builder import PROMPT_VERSION
from rag_agents.services.errors import NotFoundError, ValidationError

log = structlog.get_logger()
app = typer.Typer(no_args_is_help=True, help="Eval-харнесс: golden-датасет → метрики и отчёт")

REPORTS = Path("reports/eval")
# В diff не сравниваем «сырьё» калибровки: это не метрика качества
_NOT_COMPARED = {"max_dense_score"}


def _with_container(fn: Callable[[Container], Awaitable[None]]) -> None:
    settings = get_settings()
    configure_logging("WARNING", json=False)

    async def _run() -> None:
        c = build_container(settings)
        try:
            await fn(c)
        except (NotFoundError, ValidationError) as e:
            raise typer.BadParameter(str(e)) from e
        finally:
            await c.aclose()

    asyncio.run(_run())


def _snapshot(
    c: Container, agent: AgentOut, config: EvalConfig, judge: LLMJudge | None
) -> dict[str, Any]:
    """Всё, от чего зависит результат: без этого два прогона нельзя честно сравнить."""
    retrieval = config.retrieval or agent.settings.retrieval
    return {
        "eval": config.model_dump(mode="json"),
        "agent": {"id": str(agent.id), "slug": agent.slug, "corpus_version": agent.corpus_version},
        "retrieval": retrieval.model_dump(mode="json"),
        "generation": agent.settings.generation.model_dump(mode="json"),
        "llm": {
            "provider": c.llm.name,
            "model": c.llm.model,
            "reasoning_effort": c.llm.reasoning_effort,
        },
        "embedding_model": c.settings.embedding_model,
        "prompt_version": PROMPT_VERSION,
        "judge": f"{judge.model} {JUDGE_VERSION}" if judge else None,
        "git_sha": os.environ.get("GIT_SHA") or "unknown",
    }


def _progress(total: int) -> Callable[[EvalItemResult], None]:
    done = 0

    def on_item(it: EvalItemResult) -> None:
        nonlocal done
        done += 1
        mark = "ERR" if it.error else ("отказ" if it.refused else "ответ")
        hit = it.scores.get("hit@8")
        typer.echo(
            f"[{done:>3}/{total}] {it.item_id:<10} {it.category.value:<14} {mark:<6} "
            f"hit@8={'—' if hit is None else f'{hit:.0f}'} {it.latency_ms or 0:>6} мс"
        )

    return on_item


async def _export_langfuse(
    settings_c: Container,
    dataset: str,
    golden: list[GoldenItem],
    run: EvalRunOut,
    results: list[EvalItemResult],
) -> None:
    """Датасет и прогон в Langfuse Datasets: сравнение прогонов в UI (дублирует PG)."""
    lf = LangfuseDatasets(settings_c.settings)
    if not lf.enabled:
        return
    run_name = f"{run.config_name} {run.started_at:%Y-%m-%d %H:%M} {str(run.id)[:8]}"
    try:
        await lf.upsert_dataset(dataset, f"Golden-датасет {dataset} (ARCHITECTURE §15.1)")
        for g in golden:
            await lf.upsert_item(
                dataset,
                f"{dataset}:{g.id}",
                {"question": g.question},
                {
                    "reference_answer": g.reference_answer,
                    "key_facts": g.key_facts,
                    "expected_sources": [s.model_dump() for s in g.expected_sources],
                },
                {"category": g.category.value, "tags": g.tags},
            )
        for it in results:
            if it.trace_id:
                await lf.link_run_item(
                    run_name, f"{dataset}:{it.item_id}", it.trace_id, {"eval_run_id": str(run.id)}
                )
        await settings_c.trace.emit(
            "eval.export",
            "eval",
            "langfuse",
            f"Langfuse Datasets: датасет {dataset} ({len(golden)} вопр.) + прогон «{run_name}»",
            agent_id=run.agent_id,
        )
        typer.echo(f"Langfuse Datasets: {dataset} → прогон «{run_name}»")
    except LangfuseApiError as e:
        typer.echo(f"Langfuse Datasets: не выгружено ({e})", err=True)
    finally:
        await lf.aclose()


@app.command()
def run(
    agent: Annotated[str, typer.Option(help="slug или id агента")],
    dataset: Annotated[Path, typer.Option(help="golden JSONL")],
    config: Annotated[Path, typer.Option(help="configs/eval/<name>.yaml")],
    limit: Annotated[int, typer.Option(help="только первые N вопросов (0 — все)")] = 0,
    judge: Annotated[bool, typer.Option(help="LLM-судья (метрики генерации)")] = True,
    out_dir: Annotated[Path, typer.Option(help="куда писать отчёт")] = REPORTS,
) -> None:
    """Прогнать датасет через QueryService и записать eval_runs/eval_items + отчёт."""
    golden, sha = load_dataset(dataset)
    if limit:
        golden = golden[:limit]
    cfg = load_config(config)

    async def _run(c: Container) -> None:
        target = await c.evals.resolve_agent(agent)
        llm_judge = LLMJudge(c.llm) if judge and cfg.judge else None
        runner = EvalRunner(c.query, c.evals, c.tracer, c.query.prices, llm_judge, c.trace)
        typer.echo(f"eval: «{target.name}» × {dataset.name} ({len(golden)} вопр.) × {cfg.name}")
        result_run, results = await runner.run(
            target,
            golden,
            dataset=dataset.stem,
            dataset_sha=sha,
            config=cfg,
            snapshot=_snapshot(c, target, cfg, llm_judge),
            on_item=_progress(len(golden)),
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M")
        path = out_dir / f"{stamp}_{cfg.name}_{str(result_run.id)[:8]}.md"
        path.write_text(render_run(result_run, results), "utf-8")
        if cfg.langfuse:
            await _export_langfuse(c, dataset.stem, golden, result_run, results)
        _print_summary(result_run)
        typer.echo(f"\nпрогон: {result_run.id}\nотчёт: {path}")

    _with_container(_run)


def _print_summary(r: EvalRunOut) -> None:
    metrics: dict[str, dict[str, Any]] = (r.metrics or {}).get("metrics", {})
    typer.echo("")
    for k in ORDER:
        if k in metrics and metrics[k]["mean"] is not None:
            typer.echo(f"  {k:<20} {metrics[k]['mean']:.3f}  (n={metrics[k]['n']})")


@app.command()
def diff(
    a: Annotated[str, typer.Argument(help="id прогона A (база)")],
    b: Annotated[str, typer.Argument(help="id прогона B (эксперимент)")],
    out_dir: Annotated[Path, typer.Option(help="куда писать отчёт")] = REPORTS,
) -> None:
    """Сравнить два прогона: разница B − A с 95 % CI (парный bootstrap)."""

    async def _diff(c: Container) -> None:
        run_a, items_a = await c.evals.get_run(UUID(a))
        run_b, items_b = await c.evals.get_run(UUID(b))
        keys = sorted({k for it in items_a for k in it.scores} - _NOT_COMPARED)
        ordered = [k for k in ORDER if k in keys] + [k for k in keys if k not in ORDER]
        text = render_diff(run_a, run_b, compare(items_a, items_b, ordered))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"diff_{a[:8]}_{b[:8]}.md"
        path.write_text(text, "utf-8")
        typer.echo(text)
        typer.echo(f"отчёт: {path}")

    _with_container(_diff)


@app.command("export-langfuse")
def export_langfuse(
    run_id: Annotated[str, typer.Argument(help="id прогона")],
    datasets_dir: Annotated[Path, typer.Option(help="где лежат golden JSONL")] = Path(
        "eval/datasets"
    ),
) -> None:
    """Повторно выгрузить готовый прогон в Langfuse Datasets (если при прогоне не вышло)."""

    async def _export(c: Container) -> None:
        run_out, items = await c.evals.get_run(UUID(run_id))
        golden, sha = load_dataset(datasets_dir / f"{run_out.dataset}.jsonl")
        if sha != run_out.dataset_sha:
            typer.echo(
                f"⚠️ датасет изменился после прогона ({run_out.dataset_sha} → {sha}): "
                "эталоны в Langfuse будут из текущей версии",
                err=True,
            )
        await _export_langfuse(c, run_out.dataset, golden, run_out, items)

    _with_container(_export)


@app.command("list")
def list_runs(agent: Annotated[str, typer.Option(help="slug или id агента")]) -> None:
    """Последние прогоны агента: id, конфиг, ключевые метрики."""

    async def _list(c: Container) -> None:
        target = await c.evals.resolve_agent(agent)
        for r in await c.evals.list_runs(target.id):
            m = (r.metrics or {}).get("metrics", {})

            def g(k: str, m: dict[str, Any] = m) -> str:
                v = m.get(k, {}).get("mean")
                return "—" if v is None else f"{v:.3f}"

            state = "" if r.finished_at else " (не завершён)"
            typer.echo(
                f"{r.id}  {r.started_at:%Y-%m-%d %H:%M}  {r.config_name:<14} "
                f"hit@8={g('hit@8')} mrr={g('mrr@10')} faith={g('faithfulness')} "
                f"facts={g('key_facts_coverage')} refusal={g('refusal_correct')}{state}"
            )

    _with_container(_list)


if __name__ == "__main__":
    app()
