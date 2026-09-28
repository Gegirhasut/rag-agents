import hashlib
from uuid import UUID

from redis.asyncio import Redis

from rag_agents.core.ids import utcnow
from rag_agents.core.security import random_token
from rag_agents.domain.auth import SessionData, UserOut


def _digest(session_id: str) -> str:
    # В ключе Redis — хэш, а не сам id: дамп Redis не даёт готовых cookie для входа
    return hashlib.sha256(session_id.encode()).hexdigest()


class SessionStore:
    """Серверные сессии в Redis (ARCHITECTURE ADR-2, §11): sess:{sha256(id)}, TTL sliding.

    Индекс usess:{user_id} хранит все сессии пользователя — для «выйти везде».
    """

    def __init__(self, redis: Redis, ttl_s: int) -> None:
        self.redis = redis
        self.ttl_s = ttl_s

    @staticmethod
    def _key(session_id: str) -> str:
        return f"sess:{_digest(session_id)}"

    @staticmethod
    def _user_key(user_id: UUID) -> str:
        return f"usess:{user_id}"

    async def create(self, user: UserOut) -> tuple[str, SessionData]:
        session_id = random_token(32)
        data = SessionData(
            user_id=user.id,
            email=user.email,
            is_admin=user.is_admin,
            csrf_token=random_token(32),
            created_at=utcnow(),
        )
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.set(self._key(session_id), data.model_dump_json(), ex=self.ttl_s)
            pipe.sadd(self._user_key(user.id), _digest(session_id))
            pipe.expire(self._user_key(user.id), self.ttl_s)
            await pipe.execute()
        return session_id, data

    async def get(self, session_id: str) -> SessionData | None:
        """Чтение продлевает TTL (sliding): активный пользователь не разлогинивается."""
        raw: bytes | str | None = await self.redis.getex(self._key(session_id), ex=self.ttl_s)
        return SessionData.model_validate_json(raw) if raw else None

    async def delete(self, session_id: str, user_id: UUID) -> None:
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.delete(self._key(session_id))
            pipe.srem(self._user_key(user_id), _digest(session_id))
            await pipe.execute()

    async def delete_all(self, user_id: UUID) -> int:
        members: set[bytes | str] = await self.redis.smembers(self._user_key(user_id))
        digests = [m.decode() if isinstance(m, bytes) else m for m in members]
        if not digests:
            return 0
        await self.redis.delete(*(f"sess:{d}" for d in digests), self._user_key(user_id))
        return len(digests)
