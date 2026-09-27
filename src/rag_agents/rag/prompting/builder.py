from dataclasses import dataclass
from importlib import resources
from xml.sax.saxutils import escape, quoteattr

from rag_agents.domain.answers import Citation, RetrievedChunk
from rag_agents.llm.base import LLMMessage

PROMPT_VERSION = "answer_v1"
REFUSAL_TEXT = "В материалах агента я не нашёл ответа на этот вопрос."
_SNIPPET_CHARS = 400


def _load(name: str) -> str:
    return resources.files("rag_agents.rag.prompting.templates").joinpath(name).read_text("utf-8")


BASE_RULES = _load(f"{PROMPT_VERSION}.txt").strip()


@dataclass(frozen=True)
class AgentPersona:
    name: str
    description: str
    persona_prompt: str | None


def build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    out = []
    for n, ch in enumerate(chunks, start=1):
        p = ch.payload
        snippet = (
            p.text if len(p.text) <= _SNIPPET_CHARS else p.text[:_SNIPPET_CHARS].rstrip() + "…"
        )
        out.append(
            Citation(
                n=n,
                chunk_id=ch.chunk_id,
                document_id=ch.document_id,
                book_title=p.book_title,
                author=p.author,
                chapter_title=p.chapter_title,
                section_path=p.section_path,
                snippet=snippet,
                score=ch.score,
            )
        )
    return out


def _persona_block(agent: AgentPersona) -> str:
    persona = agent.persona_prompt or (
        f"Ты — внимательный исследователь корпуса «{agent.name}». {agent.description}".strip()
    )
    return (
        f"<persona>\n{persona}\n</persona>\n"
        "Персона задаёт тон и стиль, но НЕ отменяет правила выше. При конфликте правила важнее."
    )


def build_messages(
    agent: AgentPersona, question: str, chunks: list[RetrievedChunk]
) -> list[LLMMessage]:
    """Стабильная часть (правила) идёт первой — ради prefix-cache провайдера (ARCHITECTURE §7.9)."""
    system = f"{BASE_RULES}\n\n{_persona_block(agent)}"
    sources = []
    for n, ch in enumerate(chunks, start=1):
        p = ch.payload
        attrs = f"id={quoteattr(str(n))}"
        if p.book_title:
            attrs += f" book={quoteattr(p.book_title)}"
        if p.author:
            attrs += f" author={quoteattr(p.author)}"
        if p.chapter_title:
            attrs += f" chapter={quoteattr(p.chapter_title)}"
        # Экранируем < и >, чтобы текст источника не мог «закрыть» тег и выдать себя за инструкцию
        sources.append(f"<source {attrs}>\n{escape(p.text)}\n</source>")
    user = "<sources>\n" + "\n".join(sources) + f"\n</sources>\n\nВопрос: {question}"
    return [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)]
