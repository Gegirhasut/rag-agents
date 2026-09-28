from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agents.core.ids import utcnow, uuid7
from rag_agents.domain.auth import ApiKeyOut, ApiKeyRecord, UserCredentials, UserOut
from rag_agents.models.entities import ApiKey, User


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def get_or_create(self, email: str) -> UUID:
        uid = await self.s.scalar(select(User.id).where(User.email == email))
        if uid is not None:
            return uid
        user = User(id=uuid7(), email=email)
        self.s.add(user)
        await self.s.flush()
        return user.id

    async def get(self, user_id: UUID) -> UserOut | None:
        user = await self.s.get(User, user_id)
        return UserOut.model_validate(user) if user else None

    async def get_credentials(self, email: str) -> UserCredentials | None:
        # email — citext: сравнение без учёта регистра делает PostgreSQL
        user = await self.s.scalar(select(User).where(User.email == email))
        return UserCredentials.model_validate(user) if user else None

    async def create(self, email: str, password_hash: str, *, is_admin: bool) -> UserOut:
        user = User(id=uuid7(), email=email, password_hash=password_hash, is_admin=is_admin)
        self.s.add(user)
        await self.s.flush()
        await self.s.refresh(user)
        return UserOut.model_validate(user)

    async def update_credentials(
        self, email: str, *, password_hash: str | None = None, is_admin: bool | None = None
    ) -> UserOut | None:
        values: dict[str, object] = {}
        if password_hash is not None:
            values["password_hash"] = password_hash
        if is_admin is not None:
            values["is_admin"] = is_admin
        user = await self.s.scalar(
            update(User).where(User.email == email).values(**values).returning(User)
        )
        return UserOut.model_validate(user) if user else None


class ApiKeyRepository:
    """Ключи всегда адресуются парой (user_id, key_id): чужой ключ не отозвать и не увидеть."""

    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def create(self, user_id: UUID, name: str, prefix: str, secret_hash: bytes) -> ApiKeyOut:
        key = ApiKey(id=uuid7(), user_id=user_id, name=name, prefix=prefix, secret_hash=secret_hash)
        self.s.add(key)
        await self.s.flush()
        await self.s.refresh(key)
        return ApiKeyOut.model_validate(key)

    async def list_for_user(self, user_id: UUID) -> list[ApiKeyOut]:
        rows = await self.s.scalars(
            select(ApiKey).where(ApiKey.user_id == user_id).order_by(ApiKey.created_at.desc())
        )
        return [ApiKeyOut.model_validate(k) for k in rows]

    async def revoke(self, user_id: UUID, key_id: UUID) -> ApiKeyOut | None:
        key = await self.s.scalar(
            update(ApiKey)
            .where(ApiKey.id == key_id, ApiKey.user_id == user_id)
            # Повторный отзыв сохраняет первую дату
            .values(revoked_at=func.coalesce(ApiKey.revoked_at, utcnow()))
            .returning(ApiKey)
        )
        return ApiKeyOut.model_validate(key) if key else None

    async def find_by_prefix(self, prefix: str) -> ApiKeyRecord | None:
        key = await self.s.scalar(select(ApiKey).where(ApiKey.prefix == prefix))
        return ApiKeyRecord.model_validate(key) if key else None

    async def touch(self, key_id: UUID, now: datetime) -> None:
        await self.s.execute(update(ApiKey).where(ApiKey.id == key_id).values(last_used_at=now))
