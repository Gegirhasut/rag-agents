"""Auth: вход по паролю, серверные сессии, API-ключи (ARCHITECTURE ADR-2)."""

from datetime import timedelta
from uuid import UUID

import anyio
import structlog

from rag_agents.core.db import Database
from rag_agents.core.ids import utcnow
from rag_agents.core.security import (
    hash_password,
    new_api_key,
    parse_api_key,
    tokens_equal,
    verify_password,
)
from rag_agents.domain.auth import ApiKeyIssued, ApiKeyOut, Principal, UserOut
from rag_agents.repositories.users import ApiKeyRepository, UserRepository
from rag_agents.services.errors import NotFoundError, ValidationError
from rag_agents.services.sessions import SessionStore

log = structlog.get_logger()

# Локальный pet-проект: требований к сложности пароля нет, только непустой
MIN_PASSWORD_LEN = 1
# last_used_at пишем не чаще раза в минуту: иначе каждый запрос API — UPDATE в PG
_TOUCH_EVERY = timedelta(minutes=1)


class InvalidCredentialsError(Exception):
    """Неверный email или пароль. Наружу — одно сообщение для обоих случаев."""


class AuthService:
    def __init__(self, db: Database, sessions: SessionStore) -> None:
        self.db = db
        self.sessions = sessions

    # --- пользователи (CLI) ---

    async def create_user(self, email: str, password: str, *, is_admin: bool) -> UserOut:
        _check_password(password)
        async with self.db.uow() as uow:
            repo = UserRepository(uow.session)
            if await repo.get_credentials(email) is not None:
                raise ValidationError(f"Пользователь {email} уже существует")
            user = await repo.create(email, hash_password(password), is_admin=is_admin)
            await uow.commit()
        log.info("auth.user_created", user_id=str(user.id), admin=is_admin)
        return user

    async def set_password(
        self, email: str, password: str, *, is_admin: bool | None = None
    ) -> UserOut:
        """Смена пароля завершает все сессии пользователя."""
        _check_password(password)
        async with self.db.uow() as uow:
            user = await UserRepository(uow.session).update_credentials(
                email, password_hash=hash_password(password), is_admin=is_admin
            )
            if user is None:
                raise NotFoundError("user")
            await uow.commit()
        await self.sessions.delete_all(user.id)
        return user

    # --- web-сессии ---

    async def login(self, email: str, password: str) -> tuple[str, Principal]:
        async with self.db.session() as s:
            user = await UserRepository(s).get_credentials(email.strip())
        # argon2 — десятки-сотни мс CPU: в поток, чтобы не блокировать event loop с SSE-стримами.
        # Считается и для несуществующего email (время ответа одинаковое)
        ok = await anyio.to_thread.run_sync(
            verify_password, password, user.password_hash if user else None
        )
        if user is None or not ok or not user.is_active:
            log.info("auth.login_failed")
            raise InvalidCredentialsError
        session_id, data = await self.sessions.create(user)
        log.info("auth.login", user_id=str(user.id))
        return session_id, Principal(
            user_id=user.id,
            email=user.email,
            is_admin=user.is_admin,
            via="session",
            session_id=session_id,
            csrf_token=data.csrf_token,
        )

    async def logout(self, principal: Principal) -> None:
        if principal.session_id is not None:
            await self.sessions.delete(principal.session_id, principal.user_id)
            log.info("auth.logout", user_id=str(principal.user_id))

    async def from_session(self, session_id: str) -> Principal | None:
        data = await self.sessions.get(session_id)
        if data is None:
            return None
        return Principal(
            user_id=data.user_id,
            email=data.email,
            is_admin=data.is_admin,
            via="session",
            session_id=session_id,
            csrf_token=data.csrf_token,
        )

    # --- API-ключи ---

    async def from_api_key(self, token: str) -> Principal | None:
        parts = parse_api_key(token)
        if parts is None:
            return None
        async with self.db.uow() as uow:
            keys = ApiKeyRepository(uow.session)
            record = await keys.find_by_prefix(parts.prefix)
            if (
                record is None
                or record.revoked_at is not None
                or not tokens_equal(record.secret_hash.hex(), parts.secret_hash.hex())
            ):
                return None
            user = await UserRepository(uow.session).get(record.user_id)
            if user is None or not user.is_active:
                return None
            now = utcnow()
            if record.last_used_at is None or now - record.last_used_at > _TOUCH_EVERY:
                await keys.touch(record.id, now)
                await uow.commit()
        return Principal(user_id=user.id, email=user.email, is_admin=user.is_admin, via="api_key")

    async def issue_key(self, user_id: UUID, name: str) -> ApiKeyIssued:
        """Полный токен возвращается один раз; в БД остаются prefix и sha256(secret)."""
        parts = new_api_key()
        async with self.db.uow() as uow:
            key = await ApiKeyRepository(uow.session).create(
                user_id, name, parts.prefix, parts.secret_hash
            )
            await uow.commit()
        log.info("auth.api_key_issued", user_id=str(user_id), key_id=str(key.id))
        return ApiKeyIssued(**key.model_dump(), token=parts.token)

    async def list_keys(self, user_id: UUID) -> list[ApiKeyOut]:
        async with self.db.session() as s:
            return await ApiKeyRepository(s).list_for_user(user_id)

    async def revoke_key(self, user_id: UUID, key_id: UUID) -> ApiKeyOut:
        async with self.db.uow() as uow:
            key = await ApiKeyRepository(uow.session).revoke(user_id, key_id)
            if key is None:
                raise NotFoundError("api_key")
            await uow.commit()
        log.info("auth.api_key_revoked", user_id=str(user_id), key_id=str(key_id))
        return key


def _check_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LEN:
        raise ValidationError("Пароль не может быть пустым")
