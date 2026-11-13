from fastapi.testclient import TestClient

from rag_agents.core import i18n
from rag_agents.web.app import create_app


def test_t_defaults_to_english_and_formats() -> None:
    i18n.set_lang(None)
    assert i18n.get_lang() == "en"
    assert i18n.t("Выйти") == "Log out"
    i18n.set_lang("ru")
    assert i18n.t("Выйти") == "Выйти"
    i18n.set_lang("xx")
    assert i18n.get_lang() == "en"


def test_unknown_key_passes_through() -> None:
    i18n.set_lang("en")
    assert i18n.t("нет такого {n}", n=3) == "нет такого 3"


def test_catalog_has_no_empty_translations() -> None:
    assert all(v.strip() for v in i18n.EN.values())
    assert all(v.strip() for v in i18n.JS_EN.values())


def test_lang_route_sets_cookie_and_blocks_open_redirect() -> None:
    client = TestClient(create_app(), follow_redirects=False)
    r = client.get("/lang/ru", params={"next": "/agents/new?x=1"})
    assert r.status_code == 303
    assert r.headers["location"] == "/agents/new?x=1"
    assert "lang=ru" in r.headers["set-cookie"]
    for bad in ("https://evil.example", "//evil.example", "/\\evil.example"):
        r = client.get("/lang/en", params={"next": bad})
        assert r.headers["location"] == "/"
    r = client.get("/lang/de")
    assert "lang=en" in r.headers["set-cookie"]
