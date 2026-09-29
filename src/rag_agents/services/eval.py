"""Хранение прогонов eval (eval_runs / eval_items). Сам прогон — rag_agents.eval.runner."""

from typing import Any
from uuid import UUID

from rag_agents.core.db import Database
from rag_agents.core.ids import uuid7
from rag_agents.domain.agents import AgentOut
from rag_agents.domain.eval import EvalItemResult, EvalRunOut
from rag_agents.repositories.agents import AgentRepository
from rag_agents.repositories.eval import EvalRepository
from rag_agents.services.errors import NotFoundError, ValidationError


class EvalService:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def resolve_agent(self, ref: str) -> AgentOut:
        """Агент для админского CLI: UUID или slug (slug должен быть однозначным)."""
        async with self.db.session() as s:
            repo = AgentRepository(s)
            try:
                agent = await repo.get_any(UUID(ref))
                found = [agent] if agent and agent.deleted_at is None else []
            except ValueError:
                found = await repo.find_by_slug(ref)
        if not found:
            raise NotFoundError("agent")
        if len(found) > 1:
            ids = ", ".join(str(a.id) for a in found)
            raise ValidationError(f"slug «{ref}» есть у нескольких агентов, укажите id: {ids}")
        return found[0]

    async def start_run(
        self,
        agent_id: UUID,
        *,
        dataset: str,
        dataset_sha: str,
        config_name: str,
        config: dict[str, Any],
    ) -> EvalRunOut:
        async with self.db.uow() as uow:
            run = await EvalRepository(uow.session).create_run(
                uuid7(),
                agent_id,
                dataset=dataset,
                dataset_sha=dataset_sha,
                config_name=config_name,
                config=config,
            )
            await uow.commit()
        return run

    async def save_item(self, run_id: UUID, item: EvalItemResult) -> None:
        async with self.db.uow() as uow:
            await EvalRepository(uow.session).save_item(run_id, item)
            await uow.commit()

    async def finish_run(self, agent_id: UUID, run_id: UUID, metrics: dict[str, Any]) -> EvalRunOut:
        async with self.db.uow() as uow:
            repo = EvalRepository(uow.session)
            await repo.finish_run(run_id, metrics)
            run = await repo.get_run(agent_id, run_id)
            await uow.commit()
        if run is None:
            raise NotFoundError("eval_run")
        return run

    async def get_run(self, run_id: UUID) -> tuple[EvalRunOut, list[EvalItemResult]]:
        """Прогон с результатами по вопросам (админский CLI: diff, report)."""
        async with self.db.session() as s:
            repo = EvalRepository(s)
            run = await repo.find_run(run_id)
            if run is None:
                raise NotFoundError("eval_run")
            return run, await repo.items(run.agent_id, run_id)

    async def list_runs(self, agent_id: UUID, limit: int = 20) -> list[EvalRunOut]:
        async with self.db.session() as s:
            return await EvalRepository(s).list_runs(agent_id, limit)

    async def owned_runs(self, owner_id: UUID, agent_id: UUID, limit: int = 30) -> list[EvalRunOut]:
        """Прогоны агента для страницы «Качество»: чужой агент → NotFoundError (404)."""
        async with self.db.session() as s:
            if await AgentRepository(s).get(owner_id, agent_id) is None:
                raise NotFoundError("agent")
            return await EvalRepository(s).list_runs(agent_id, limit)

    async def owned_run(
        self, owner_id: UUID, agent_id: UUID, run_id: UUID
    ) -> tuple[EvalRunOut, list[EvalItemResult]]:
        """Прогон с вопросами; прогон другого агента (даже своего владельца) → NotFoundError."""
        async with self.db.session() as s:
            if await AgentRepository(s).get(owner_id, agent_id) is None:
                raise NotFoundError("agent")
            repo = EvalRepository(s)
            run = await repo.get_run(agent_id, run_id)
            if run is None:
                raise NotFoundError("eval_run")
            return run, await repo.items(agent_id, run_id)
