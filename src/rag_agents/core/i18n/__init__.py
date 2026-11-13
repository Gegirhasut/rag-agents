"""Простая i18n без зависимостей: ключ — русская строка, каталог EN — перевод.

Язык запроса кладёт middleware в contextvar (cookie `lang`, по умолчанию en).
`t("Строка {n}", n=1)` — перевод + str.format. Неизвестный ключ возвращается как есть.
"""

from contextvars import ContextVar
from typing import Any

from rag_agents.core.i18n import catalog_core, catalog_eval, catalog_insights, catalog_system

LANGS = ("en", "ru")
DEFAULT_LANG = "en"
COOKIE = "lang"

EN: dict[str, str] = {
    **catalog_core.EN,
    **catalog_eval.EN,
    **catalog_insights.EN,
    **catalog_system.EN,
}
# Строки, которые нужны JS (static/*.js): отдаются в base.html как window.I18N
JS_EN: dict[str, str] = {
    **catalog_core.JS_EN,
    **catalog_eval.JS_EN,
    **catalog_insights.JS_EN,
    **catalog_system.JS_EN,
}

_lang: ContextVar[str] = ContextVar("lang", default=DEFAULT_LANG)


def normalize(code: str | None) -> str:
    return code if code in LANGS else DEFAULT_LANG


def set_lang(code: str | None) -> None:
    _lang.set(normalize(code))


def get_lang() -> str:
    return _lang.get()


def t(text: str, /, **kwargs: Any) -> str:
    """kwargs["ctx"] различает омонимы: ключ каталога `ctx|текст` («Вход» — логин и токены)."""
    ctx = kwargs.pop("ctx", None)
    key = f"{ctx}|{text}" if ctx else text
    s = EN.get(key, text) if _lang.get() == "en" else text
    return s.format(**kwargs) if kwargs else s


def js_catalog() -> dict[str, str]:
    return JS_EN if _lang.get() == "en" else {}
