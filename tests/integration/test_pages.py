"""Каждая страница и фрагмент UI рендерятся (200) на «грязных» данных: старые форматы usage,
отказы, ошибки, незавершённые ответы, failed-документы. Ловит 500 из шаблонов."""

from collections.abc import Callable
from uuid import UUID

import pytest

from rag_agents.domain.agents import AgentCreate
from rag_agents.domain.enums import MessageRole, MessageStatus
from rag_agents.repositories.chats import ChatRepository
from rag_agents.repositories.documents import DocumentRepository
from tests.integration.conftest import Session, Stack

pytestmark = pytest.mark.integration
Browser = Callable[[], Session]

# usage ответов итерации 1 — до полей cost_*, t_embed_ms, t_search_ms (мини-итерация 1.5)
USAGE_ITER1 = {
    "provider": "deepseek",
    "model": "deepseek-flash",
    "reasoning_effort": "low",
    "input_tokens": 2400,
    "output_tokens": 700,
    "reasoning_tokens": 60,
    "cached_input_tokens": 0,
    "t_retrieval_ms": 800,
    "t_first_token_ms": 2100,
    "t_total_ms": 5000,
}
USAGE_CURRENT = {
    **USAGE_ITER1,
    "t_embed_ms": 300,
    "t_search_ms": 500,
    "cost_usd": 0.0008,
    "cost_peak": False,
}
USAGE_REFUSAL = {**USAGE_CURRENT, "model": "none", "provider": "none", "cost_usd": 0.0}


@pytest.fixture(scope="module")
async def rich_agent(stack: Stack) -> tuple[str, UUID]:
    """Агент со всеми видами сообщений и документов, которые встречаются в живой БД."""
    c = stack.container
    user = await stack.user(admin=True)
    agent = await c.agents.create(user.id, AgentCreate(name="Грязные данные"))
    ok_doc, _ = await c.documents.upload(user.id, agent.id, "ok.txt", _one(b"ok"))
    bad_doc, _ = await c.documents.upload(user.id, agent.id, "bad.txt", _one(b"bad"))
    async with c.db.uow() as uow:
        docs = DocumentRepository(uow.session)
        await docs.mark_done(ok_doc.id, {"parse_ms": 5})
        await docs.mark_failed(bad_doc.id, "empty_document", "В файле нет текста")
        chats = ChatRepository(uow.session)
        chat = await chats.create_chat(agent.id, user.id, "q")
        cases: list[
            tuple[
                MessageStatus, dict[str, object] | None, bool | None, list[dict[str, object]] | None
            ]
        ] = [
            (MessageStatus.DONE, USAGE_ITER1, False, [_citation()]),
            (MessageStatus.DONE, USAGE_CURRENT, False, [_citation()]),
            (MessageStatus.DONE, USAGE_REFUSAL, True, []),
            (MessageStatus.DONE, None, None, None),
            (MessageStatus.ERROR, None, None, None),
            (MessageStatus.CANCELLED, None, None, None),
            (MessageStatus.PENDING, None, None, None),
        ]
        for status, usage, refused, citations in cases:
            await chats.add_message(chat.id, MessageRole.USER, MessageStatus.DONE, "вопрос")
            msg = await chats.add_message(chat.id, MessageRole.ASSISTANT, MessageStatus.PENDING)
            await chats.finish_message(
                msg.id,
                status=status,
                content="Ответ [1]." if status == MessageStatus.DONE else "",
                citations=citations,
                refused=refused,
                usage=usage,
                trace_id="0" * 32,
            )
        await uow.commit()
    return user.email, agent.id


def _one(data: bytes):  # type: ignore[no-untyped-def]
    async def gen():  # type: ignore[no-untyped-def]
        yield data

    return gen()


def _citation() -> dict[str, object]:
    return {
        "n": 1,
        "chunk_id": "01a0d517-fde0-73e0-8da3-f88dc859f9a4",
        "document_id": "01a0d517-fde0-73e0-8da3-f88dc859f9a5",
        "book_title": "Исповедь",
        "author": "Л. Н. Толстой",
        "chapter_title": "IV",
        "section_path": ["IV"],
        "snippet": "фрагмент",
        "score": 0.71,
    }


PAGES = [
    "/",
    "/agents/new",
    "/agents/{agent}",
    "/agents/{agent}/documents/status",
    "/settings/api-keys",
    "/insights",
    "/insights/overview?period=24h",
    "/insights/overview?period=30d",
    "/system",
    "/system?agent={agent}",
    "/system/agents/{agent}/vector-map",
]


@pytest.mark.parametrize("path", PAGES)
async def test_page_renders_on_real_world_data(
    browser: Browser, rich_agent: tuple[str, UUID], path: str
) -> None:
    email, agent_id = rich_agent
    b = browser()
    await b.login(email)
    url = path.format(agent=agent_id)
    r = await b.http.get(url, headers={"HX-Request": "true"} if "/status" in path else None)
    assert r.status_code in (200, 286), f"{url} → {r.status_code}: {r.text[:300]}"


async def test_agent_page_shows_old_and_new_usage(
    browser: Browser, rich_agent: tuple[str, UUID]
) -> None:
    email, agent_id = rich_agent
    b = browser()
    await b.login(email)
    html = (await b.http.get(f"/agents/{agent_id}")).text
    assert "первый токен 2100 мс" in html  # старый формат usage
    assert "$0.0008" in html  # новый — со стоимостью
    assert "ответ без вызова LLM" in html  # отказ до LLM
