import re
from uuid import UUID

import structlog

from rag_agents.core.config import Settings
from rag_agents.core.db import Database
from rag_agents.domain.agents import (
    AgentCreate,
    AgentListItem,
    AgentOut,
    AgentSettings,
    AgentUpdate,
)
from rag_agents.rag.index.qdrant import QdrantChunkIndex, collection_name
from rag_agents.repositories.agents import AgentRepository
from rag_agents.repositories.users import UserRepository
from rag_agents.services.errors import NotFoundError

log = structlog.get_logger()

_LATIN = "a b v g d e e zh z i i k l m n o p r s t u f h ts ch sh sch - y - e yu ya".split()  # noqa: SIM905
_TRANSLIT = {
    c: ("" if lat == "-" else lat)
    for c, lat in zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя", _LATIN, strict=True)
}


def slugify(name: str) -> str:
    s = "".join(_TRANSLIT.get(c, c) for c in name.lower())
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:60] or "agent"


class AgentService:
    def __init__(self, db: Database, index: QdrantChunkIndex, settings: Settings) -> None:
        self.db = db
        self.index = index
        self.settings = settings

    async def ensure_user(self, email: str) -> UUID:
        async with self.db.uow() as uow:
            uid = await UserRepository(uow.session).get_or_create(email)
            await uow.commit()
            return uid

    async def list(self, owner_id: UUID) -> list[AgentListItem]:
        async with self.db.session() as s:
            return await AgentRepository(s).list_for_owner(owner_id)

    async def get(self, owner_id: UUID, agent_id: UUID) -> AgentOut:
        async with self.db.session() as s:
            agent = await AgentRepository(s).get(owner_id, agent_id)
        if agent is None:
            raise NotFoundError("agent")
        return agent

    async def create(self, owner_id: UUID, data: AgentCreate) -> AgentOut:
        st = self.settings
        collection = collection_name(
            st.qdrant_collection_prefix, st.embedding_model, st.embedding_dim
        )
        # Коллекция общая для всех агентов с этой моделью; создание идемпотентно
        await self.index.ensure_collection(collection, st.embedding_dim)
        async with self.db.uow() as uow:
            repo = AgentRepository(uow.session)
            base = slugify(data.name)
            slug, n = base, 1
            while await repo.slug_taken(owner_id, slug):
                n += 1
                slug = f"{base}-{n}"
            agent = await repo.create_with_index(
                owner_id,
                name=data.name,
                slug=slug,
                description=data.description,
                persona_prompt=data.persona_prompt or None,
                settings=AgentSettings(),
                embedding_model=st.embedding_model,
                dim=st.embedding_dim,
                collection=collection,
            )
            await uow.commit()
        return agent

    async def update(self, owner_id: UUID, agent_id: UUID, data: AgentUpdate) -> AgentOut:
        """Частичное обновление: меняются только переданные поля (slug не меняется)."""
        values: dict[str, object] = {}
        for field in data.model_fields_set:
            value = getattr(data, field)
            if field == "name" and value is None:
                continue  # имя обязательно: null в PATCH означает «не менять»
            if field == "description":
                value = value or ""
            if field == "persona_prompt":
                value = value or None
            values[field] = value
        async with self.db.uow() as uow:
            repo = AgentRepository(uow.session)
            agent = (
                await repo.update_fields(owner_id, agent_id, values)
                if values
                else await repo.get(owner_id, agent_id)
            )
            if agent is None:
                raise NotFoundError("agent")
            await uow.commit()
        return agent

    async def delete(self, owner_id: UUID, agent_id: UUID) -> None:
        """Мягкое удаление: агент сразу пропадает из всех чтений (и API, и web)."""
        async with self.db.uow() as uow:
            if not await AgentRepository(uow.session).soft_delete(owner_id, agent_id):
                raise NotFoundError("agent")
            await uow.commit()
        log.info("agent.deleted", agent_id=str(agent_id))
