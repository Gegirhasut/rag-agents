import pytest

from rag_agents.core.security import (
    hash_api_secret,
    hash_password,
    new_api_key,
    parse_api_key,
    verify_password,
)
from rag_agents.web.routes.auth import safe_next


def test_password_hash_is_argon2id_and_verifies() -> None:
    h = hash_password("correct horse")
    assert h.startswith("$argon2id$")
    assert verify_password("correct horse", h)
    assert not verify_password("wrong horse", h)


def test_missing_hash_never_verifies() -> None:
    assert not verify_password("anything", None)


def test_api_key_roundtrip_stores_only_hash() -> None:
    key = new_api_key()
    parsed = parse_api_key(key.token)
    assert parsed == key
    assert key.token.startswith(f"rag_{key.prefix}_")
    assert len(key.prefix) == 8
    assert key.secret not in key.secret_hash.hex()
    assert hash_api_secret(parsed.secret) == key.secret_hash


@pytest.mark.parametrize(
    "token",
    ["", "rag_", "rag_zzzzzzzz_" + "a" * 43, "sk_12345678_" + "a" * 43, "rag_1234abcd_short"],
)
def test_malformed_api_keys_are_rejected(token: str) -> None:
    assert parse_api_key(token) is None


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("/agents/1", "/agents/1"),
        ("/insights?period=7d", "/insights?period=7d"),
        (None, "/"),
        ("https://evil.example", "/"),
        ("//evil.example", "/"),
        ("/\\evil.example", "/"),
    ],
)
def test_login_redirect_is_local_only(target: str | None, expected: str) -> None:
    assert safe_next(target) == expected
