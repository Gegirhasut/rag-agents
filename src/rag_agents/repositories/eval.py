from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agents.core.ids import utcnow
from rag_agents.domain.eval import EvalItemResult, EvalRunOut
from rag_agents.models.entities import EvalItem, EvalRun


class EvalRepository:
    """eval_runs / eval_items. Прогон всегда привязан к агенту: чтение — с agent_id."""

    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def create_run(
        self,
        run_id: UUID,
        agent_id: UUID,
        *,
        dataset: str,
        dataset_sha: str,
        config_name: str,
        config: dict[str, Any],
    ) -> EvalRunOut:
        run = EvalRun(
            id=run_id,
            agent_id=agent_id,
            dataset=dataset,
            dataset_sha=dataset_sha,
            config_name=config_name,
            config=config,
            started_at=utcnow(),
        )
        self.s.add(run)
        await self.s.flush()
        return EvalRunOut.model_validate(run)

    async def save_item(self, run_id: UUID, item: EvalItemResult) -> None:
        """Upsert: перезапуск вопроса в том же прогоне перезаписывает результат."""
        values = {
            "run_id": run_id,
            "item_id": item.item_id,
            "question": item.question,
            "category": item.category.value,
            "retrieved": [r.model_dump(mode="json") for r in item.retrieved],
            "answer": item.answer,
            "error": item.error,
            "refused": item.refused,
            "scores": item.scores,
            "judge": item.judge,
            "trace_id": item.trace_id,
            "latency_ms": item.latency_ms,
            "ttft_ms": item.ttft_ms,
            "cost_usd": None if item.cost_usd is None else Decimal(str(round(item.cost_usd, 6))),
        }
        stmt = insert(EvalItem).values(**values)
        keys = {"run_id", "item_id"}
        await self.s.execute(
            stmt.on_conflict_do_update(
                index_elements=["run_id", "item_id"],
                set_={k: stmt.excluded[k] for k in values if k not in keys},
            )
        )

    async def finish_run(self, run_id: UUID, metrics: dict[str, Any]) -> None:
        await self.s.execute(
            update(EvalRun)
            .where(EvalRun.id == run_id)
            .values(metrics=metrics, finished_at=utcnow())
        )

    async def get_run(self, agent_id: UUID, run_id: UUID) -> EvalRunOut | None:
        run = await self.s.scalar(
            select(EvalRun).where(EvalRun.id == run_id, EvalRun.agent_id == agent_id)
        )
        return EvalRunOut.model_validate(run) if run else None

    async def find_run(self, run_id: UUID) -> EvalRunOut | None:
        """Админский CLI (eval diff): прогон по id без владельца."""
        run = await self.s.get(EvalRun, run_id)
        return EvalRunOut.model_validate(run) if run else None

    async def list_runs(self, agent_id: UUID, limit: int = 20) -> list[EvalRunOut]:
        rows = await self.s.scalars(
            select(EvalRun)
            .where(EvalRun.agent_id == agent_id)
            .order_by(EvalRun.started_at.desc())
            .limit(limit)
        )
        return [EvalRunOut.model_validate(r) for r in rows]

    async def items(self, agent_id: UUID, run_id: UUID) -> list[EvalItemResult]:
        rows = await self.s.scalars(
            select(EvalItem)
            .join(EvalRun, EvalRun.id == EvalItem.run_id)
            .where(EvalItem.run_id == run_id, EvalRun.agent_id == agent_id)
            .order_by(EvalItem.item_id)
        )
        return [
            EvalItemResult.model_validate(
                {
                    "item_id": r.item_id,
                    "question": r.question,
                    "category": r.category,
                    "answer": r.answer,
                    "error": r.error,
                    "refused": r.refused,
                    "retrieved": r.retrieved or [],
                    "scores": r.scores or {},
                    "judge": r.judge,
                    "trace_id": r.trace_id,
                    "latency_ms": r.latency_ms,
                    "ttft_ms": r.ttft_ms,
                    "cost_usd": None if r.cost_usd is None else float(r.cost_usd),
                }
            )
            for r in rows
        ]
