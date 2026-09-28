from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StringConstraints


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    is_admin: bool
    is_active: bool
    created_at: datetime


class UserCredentials(UserOut):
    """Только для AuthService: хэш пароля наружу из сервиса не выходит."""

    password_hash: str | None


class Principal(BaseModel):
    """Кто делает запрос: пользователь из cookie-сессии (web) или по API-ключу (Bearer)."""

    model_config = ConfigDict(frozen=True)

    user_id: UUID
    email: str
    is_admin: bool
    via: Literal["session", "api_key"]
    session_id: str | None = None
    csrf_token: str | None = None


class SessionData(BaseModel):
    """Значение sess:{id} в Redis (ARCHITECTURE §11)."""

    user_id: UUID
    email: str
    is_admin: bool
    csrf_token: str
    created_at: datetime


class ApiKeyCreate(BaseModel):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]


class ApiKeyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    prefix: str
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None

    @property
    def active(self) -> bool:
        return self.revoked_at is None


class ApiKeyIssued(ApiKeyOut):
    """Ответ на выпуск ключа: полный токен виден только здесь, один раз."""

    token: str


class ApiKeyRecord(BaseModel):
    """Строка для проверки Bearer-токена (секрет — только sha256)."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    secret_hash: bytes
    last_used_at: datetime | None
    revoked_at: datetime | None
