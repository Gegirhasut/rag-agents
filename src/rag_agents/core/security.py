"""Криптопримитивы auth: пароли (argon2id), API-ключи, случайные токены (ARCHITECTURE ADR-2)."""

import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass

from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

# Параметры argon2-cffi по умолчанию (RFC 9106, профиль «low memory»): 64 МиБ, t=3, p=4.
# ~50 мс на хэш на CPU VM — приемлемо для логина и не даёт перебирать пароли быстро.
_hasher = PasswordHash((Argon2Hasher(),))
# Хэш-заглушка: проверяем пароль и для несуществующего email, чтобы время ответа не выдавало,
# есть ли такой пользователь
_DUMMY_HASH = _hasher.hash(secrets.token_urlsafe(16))

API_KEY_PREFIX = "rag"
_API_KEY_RE = re.compile(r"^rag_([0-9a-f]{8})_([A-Za-z0-9_-]{32,64})$")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    """Постоянное время для отсутствующего хэша: всегда считаем argon2."""
    if password_hash is None:
        _hasher.verify(password, _DUMMY_HASH)
        return False
    return _hasher.verify(password, password_hash)


def random_token(nbytes: int = 32) -> str:
    return secrets.token_urlsafe(nbytes)


def tokens_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


@dataclass(frozen=True)
class ApiKeyParts:
    prefix: str
    secret: str

    @property
    def token(self) -> str:
        return f"{API_KEY_PREFIX}_{self.prefix}_{self.secret}"

    @property
    def secret_hash(self) -> bytes:
        return hash_api_secret(self.secret)


def new_api_key() -> ApiKeyParts:
    """rag_<prefix>_<secret>: prefix (8 hex) ищется в БД, secret хранится только как sha256.

    У секрета 256 бит энтропии, поэтому медленный хэш не нужен: перебор sha256 по такому
    пространству невозможен, а проверка на каждом запросе API остаётся дешёвой.
    """
    return ApiKeyParts(prefix=secrets.token_hex(4), secret=secrets.token_urlsafe(32))


def parse_api_key(token: str) -> ApiKeyParts | None:
    m = _API_KEY_RE.match(token.strip())
    return ApiKeyParts(prefix=m.group(1), secret=m.group(2)) if m else None


def hash_api_secret(secret: str) -> bytes:
    return hashlib.sha256(secret.encode()).digest()
