"""LLM-судья: метрики генерации в духе RAGAS, своей реализацией (ARCHITECTURE §15.2, ADR-10).

Три вызова на вопрос, каждый возвращает один JSON-объект (response_format=json_object):
1. faithfulness — ответ раскладывается на утверждения, каждое сверяется с источниками;
   заодно проверяется, подтверждает ли процитированный [n] своё утверждение (citation support);
2. context — какие источники полезны для эталонного ответа (context precision как AP)
   и какие ключевые факты в них есть (context recall);
3. answer — отвечает ли ответ на вопрос (relevancy) и какие ключевые факты он покрывает.

Ошибка судьи (сеть, невалидный JSON) не валит прогон: метрика вопроса = None, причина — в raw.
"""

import json
from dataclasses import dataclass, field
from importlib import resources
from typing import Any
from xml.sax.saxutils import escape

import structlog

from rag_agents.domain.answers import RetrievedChunk
from rag_agents.domain.eval import GoldenItem
from rag_agents.eval.metrics import average_precision
from rag_agents.llm.base import LLMError, LLMMessage, LLMProvider, LLMRequest, LLMUsage, complete

log = structlog.get_logger()

JUDGE_VERSION = "judge_v1"


def _load(step: str) -> str:
    """Промпты судьи версионируются, как промпты ответа: смена текста = новая JUDGE_VERSION."""
    path = resources.files("rag_agents.eval.prompts").joinpath(f"{JUDGE_VERSION}_{step}.txt")
    return path.read_text("utf-8")


_MAX_CLAIMS = 12

_SYSTEM = (
    "Ты — строгий и беспристрастный эксперт по оценке ответов RAG-системы по русской литературе. "
    "Оценивай только по предоставленным материалам, не используй собственные знания о книгах. "
    "Отвечай ОДНИМ JSON-объектом без markdown и пояснений вне JSON."
)

_FAITHFULNESS = _load("faithfulness")

_CONTEXT = _load("context")

_ANSWER = _load("answer")


class JudgeFormatError(ValueError):
    pass


@dataclass
class JudgeResult:
    scores: dict[str, float | None] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    usage: list[LLMUsage] = field(default_factory=list)


def _sources(chunks: list[RetrievedChunk]) -> str:
    return "\n".join(
        f'<source id="{n}">\n{escape(c.payload.text)}\n</source>'
        for n, c in enumerate(chunks, start=1)
    )


def _facts(item: GoldenItem) -> list[str]:
    # Без key_facts эталонный ответ целиком считается одним фактом
    return item.key_facts or [item.reference_answer]


def _bools(value: Any, n: int, key: str) -> list[bool]:
    if not isinstance(value, list) or len(value) != n:
        raise JudgeFormatError(f"{key}: ожидался список из {n} bool, пришло {value!r}"[:200])
    return [v is True for v in value]


def faithfulness_scores(data: dict[str, Any]) -> dict[str, float | None]:
    claims = data.get("claims")
    if not isinstance(claims, list):
        raise JudgeFormatError("claims: ожидался список")
    if not claims:
        return {"faithfulness": None, "citation_support": None}
    supported = sum(1 for c in claims if c.get("supported") is True)
    cited = [c for c in claims if c.get("cited")]
    ok = sum(1 for c in cited if c.get("citation_ok") is True)
    return {
        "faithfulness": supported / len(claims),
        "citation_support": ok / len(cited) if cited else None,
    }


def context_scores(data: dict[str, Any], n_sources: int, n_facts: int) -> dict[str, float | None]:
    useful = data.get("useful")
    if not isinstance(useful, list):
        raise JudgeFormatError("useful: ожидался список номеров")
    wanted = {int(n) for n in useful if isinstance(n, int | float)}
    relevance = [n in wanted for n in range(1, n_sources + 1)]
    facts = _bools(data.get("facts_in_sources"), n_facts, "facts_in_sources")
    return {
        "context_precision": average_precision(relevance),
        "context_recall": sum(facts) / n_facts,
    }


def answer_scores(data: dict[str, Any], n_facts: int) -> dict[str, float | None]:
    relevancy = data.get("relevancy")
    if not isinstance(relevancy, int | float) or not 0 <= relevancy <= 1:
        raise JudgeFormatError(f"relevancy: {relevancy!r}")
    facts = _bools(data.get("facts_covered"), n_facts, "facts_covered")
    return {"answer_relevancy": float(relevancy), "key_facts_coverage": sum(facts) / n_facts}


_ATTEMPTS = 2


class LLMJudge:
    # deepseek-flash рассуждает даже на effort=low. Замер 2026-09-29: faithfulness длинного
    # ответа (1–2 тыс. символов) — 5 500–14 300 токенов reasoning; при меньшем лимите content
    # пуст. Платим за фактически потраченные токены, поэтому запас почти бесплатен
    def __init__(self, llm: LLMProvider, *, max_tokens: int = 24_000) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    @property
    def model(self) -> str:
        return f"{self.llm.name}/{self.llm.model}"

    async def _ask(self, prompt: str, out: JudgeResult, key: str) -> dict[str, Any] | None:
        req = LLMRequest(
            messages=[
                LLMMessage(role="system", content=_SYSTEM),
                LLMMessage(role="user", content=prompt),
            ],
            temperature=0.0,
            max_tokens=self.max_tokens,
            purpose="judge",
            json_mode=True,
        )
        # Длина рассуждений на одном и том же шаге заметно гуляет (на wp-003: ~3000 токенов
        # в двух попытках из трёх, весь лимит в третьей), поэтому после обрезки — один повтор
        for attempt in range(1, _ATTEMPTS + 1):
            try:
                done = await complete(self.llm, req)
            except LLMError as e:
                out.raw[key] = {"error": f"llm: {e}"[:300]}
                return None
            out.usage.append(done.usage)
            if done.finish_reason != "length":
                break
            log.warning(
                "eval.judge_truncated", step=key, attempt=attempt, max_tokens=self.max_tokens
            )
        else:
            # Обрезанный JSON не разбирается, а «invalid json» скрыл бы причину
            out.raw[key] = {"error": "truncated", "reasoning_tokens": done.usage.reasoning_tokens}
            return None
        try:
            data = json.loads(done.text)
        except json.JSONDecodeError:
            out.raw[key] = {"error": "invalid json", "text": done.text[:300]}
            return None
        if not isinstance(data, dict):
            out.raw[key] = {"error": "not an object", "text": done.text[:300]}
            return None
        out.raw[key] = data
        return data

    async def evaluate(
        self,
        item: GoldenItem,
        answer: str | None,
        refused: bool,
        context: list[RetrievedChunk],
    ) -> JudgeResult:
        """Метрики генерации для вопроса с ответом в корпусе.

        Отказ или пустой ответ на такой вопрос — ошибка системы: relevancy и покрытие фактов
        = 0, faithfulness не определена (утверждений нет). Контекст оценивается в любом случае.
        """
        out = JudgeResult()
        facts = _facts(item)
        fact_lines = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, start=1))
        sources = _sources(context)
        steps: list[tuple[str, str, Any]] = []
        if context:
            steps.append(
                (
                    "context",
                    _CONTEXT.format(
                        question=escape(item.question),
                        reference=escape(item.reference_answer),
                        facts=escape(fact_lines),
                        sources=sources,
                    ),
                    lambda d: context_scores(d, len(context), len(facts)),
                )
            )
        if answer and answer.strip() and not refused:
            steps.append(
                (
                    "faithfulness",
                    _FAITHFULNESS.format(
                        max_claims=_MAX_CLAIMS, sources=sources, answer=escape(answer)
                    ),
                    faithfulness_scores,
                )
            )
            steps.append(
                (
                    "answer",
                    _ANSWER.format(
                        question=escape(item.question),
                        reference=escape(item.reference_answer),
                        facts=escape(fact_lines),
                        answer=escape(answer),
                    ),
                    lambda d: answer_scores(d, len(facts)),
                )
            )
        else:
            # Отказ или пустой ответ (reasoning съел max_output_tokens) — провал для вопроса
            # из корпуса. Без нулей такие вопросы выпали бы из среднего и завысили его
            out.scores.update({"answer_relevancy": 0.0, "key_facts_coverage": 0.0})
        for key, prompt, score in steps:
            data = await self._ask(prompt, out, key)
            if data is None:
                continue
            try:
                out.scores.update(score(data))
            except (JudgeFormatError, TypeError, ValueError, AttributeError) as e:
                out.raw[key] = {"error": f"format: {e}"[:300], "data": data}
                log.warning("eval.judge_format", item_id=item.id, step=key)
        return out
