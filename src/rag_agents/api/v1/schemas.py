"""Формы ответов JSON API поверх domain-DTO (ARCHITECTURE §6.2)."""

from pydantic import BaseModel

from rag_agents.domain.agents import AgentOut


class AgentItem(AgentOut):
    documents_total: int
    documents_done: int


class Page[T](BaseModel):
    """Пагинации пока нет (агентов у пользователя единицы): total = len(items)."""

    items: list[T]
    total: int


class Problem(BaseModel):
    """RFC 9457: тело любой ошибки API (application/problem+json)."""

    type: str
    title: str
    status: int
    detail: str
    code: str


ERRORS = {
    401: {"model": Problem, "description": "Нет или неверный API-ключ"},
    404: {"model": Problem, "description": "Не найдено или принадлежит другому пользователю"},
    422: {"model": Problem, "description": "Некорректный запрос"},
}
