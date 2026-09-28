from uuid import UUID

from sqlalchemy import case, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agents.core.ids import utcnow, uuid7
from rag_agents.domain.agents import AgentIndexOut, AgentListItem, AgentOut, AgentSettings
from rag_agents.domain.enums import DocumentStatus, IndexStatus
from rag_agents.models.entities import Agent, AgentIndex, Document


def _agent_out(a: Agent) -> AgentOut:
    return AgentOut(
        id=a.id,
        owner_id=a.owner_id,
        name=a.name,
        slug=a.slug,
        description=a.description,
        persona_prompt=a.persona_prompt,
        settings=AgentSettings.model_validate(a.settings or {}),
        active_index_id=a.active_index_id,
        corpus_version=a.corpus_version,
        created_at=a.created_at,
        deleted_at=a.deleted_at,
    )


class AgentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def list_for_owner(self, owner_id: UUID) -> list[AgentListItem]:
        done = func.count(case((Document.status == DocumentStatus.DONE, 1)))
        stmt = (
            select(Agent, func.count(Document.id), done)
            .outerjoin(Document, Document.agent_id == Agent.id)
            .where(Agent.owner_id == owner_id, Agent.deleted_at.is_(None))
            .group_by(Agent.id)
            .order_by(Agent.created_at.desc())
        )
        rows = (await self.s.execute(stmt)).all()
        return [
            AgentListItem(agent=_agent_out(a), documents_total=total, documents_done=d)
            for a, total, d in rows
        ]

    async def get(self, owner_id: UUID, agent_id: UUID) -> AgentOut | None:
        a = await self.s.scalar(
            select(Agent).where(
                Agent.id == agent_id, Agent.owner_id == owner_id, Agent.deleted_at.is_(None)
            )
        )
        return _agent_out(a) if a else None

    async def slug_taken(self, owner_id: UUID, slug: str) -> bool:
        found = await self.s.scalar(
            select(Agent.id).where(Agent.owner_id == owner_id, Agent.slug == slug)
        )
        return found is not None

    async def create_with_index(
        self,
        owner_id: UUID,
        *,
        name: str,
        slug: str,
        description: str,
        persona_prompt: str | None,
        settings: AgentSettings,
        embedding_model: str,
        dim: int,
        collection: str,
    ) -> AgentOut:
        agent_id, index_id = uuid7(), uuid7()
        agent = Agent(
            id=agent_id,
            owner_id=owner_id,
            name=name,
            slug=slug,
            description=description,
            persona_prompt=persona_prompt,
            settings=settings.model_dump(),
            active_index_id=index_id,  # FK отложенный — индекс вставляется в той же транзакции
            corpus_version=0,
        )
        index = AgentIndex(
            id=index_id,
            agent_id=agent_id,
            embedding_model=embedding_model,
            dim=dim,
            collection=collection,
            status=IndexStatus.ACTIVE,
            activated_at=utcnow(),
        )
        self.s.add_all([agent, index])
        await self.s.flush()
        await self.s.refresh(agent)
        return _agent_out(agent)

    async def get_index(self, agent_id: UUID, index_id: UUID) -> AgentIndexOut | None:
        idx = await self.s.scalar(
            select(AgentIndex).where(AgentIndex.id == index_id, AgentIndex.agent_id == agent_id)
        )
        return AgentIndexOut.model_validate(idx) if idx else None

    async def get_unscoped(self, agent_id: UUID) -> AgentOut | None:
        """Для воркера: владелец уже проверен при постановке задачи."""
        a = await self.s.get(Agent, agent_id)
        return _agent_out(a) if a and a.deleted_at is None else None

    async def bump_corpus_version(self, agent_id: UUID) -> None:
        agent = await self.s.get(Agent, agent_id, with_for_update=True)
        if agent is not None:
            agent.corpus_version += 1

    async def update_fields(
        self, owner_id: UUID, agent_id: UUID, values: dict[str, object]
    ) -> AgentOut | None:
        a = await self.s.scalar(
            update(Agent)
            .where(Agent.id == agent_id, Agent.owner_id == owner_id, Agent.deleted_at.is_(None))
            .values(**values, updated_at=func.now())
            .returning(Agent)
        )
        return _agent_out(a) if a else None

    async def soft_delete(self, owner_id: UUID, agent_id: UUID) -> bool:
        """Агент исчезает для всех чтений сразу; точки в Qdrant и файлы удалит задача
        очистки (итерация 3)."""
        found = await self.s.scalar(
            update(Agent)
            .where(Agent.id == agent_id, Agent.owner_id == owner_id, Agent.deleted_at.is_(None))
            .values(deleted_at=utcnow())
            .returning(Agent.id)
        )
        return found is not None

    async def get_any(self, agent_id: UUID) -> AgentOut | None:
        """Включая мягко удалённых: для задачи очистки."""
        a = await self.s.get(Agent, agent_id)
        return _agent_out(a) if a else None

    async def list_indexes(self, agent_id: UUID) -> list[AgentIndexOut]:
        rows = await self.s.scalars(select(AgentIndex).where(AgentIndex.agent_id == agent_id))
        return [AgentIndexOut.model_validate(i) for i in rows]
