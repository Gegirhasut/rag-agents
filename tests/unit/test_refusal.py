import pytest

from rag_agents.rag.prompting.builder import BASE_RULES, PROMPT_VERSION
from rag_agents.rag.prompting.refusal import (
    NO_ANSWER_MARKER,
    REFUSAL_TEXTS,
    MarkerFilter,
    detect_language,
    refusal_text,
    strip_marker,
)


def run(deltas: list[str]) -> tuple[list[str], MarkerFilter]:
    f = MarkerFilter()
    out = [f.feed(d) for d in deltas]
    out.append(f.flush())
    return [o for o in out if o], f


def test_prompt_asks_for_the_marker() -> None:
    assert PROMPT_VERSION == "answer_v2"
    assert NO_ANSWER_MARKER in BASE_RULES


def test_marker_in_one_token_is_stripped_and_sets_refused() -> None:
    out, f = run([f"{NO_ANSWER_MARKER} I could not find it."])
    assert "".join(out) == "I could not find it."
    assert f.refused is True
    assert f.text == "I could not find it."
    assert f.raw == f"{NO_ANSWER_MARKER} I could not find it."


@pytest.mark.parametrize(
    "deltas",
    [
        ["[[", "NO_AN", "SWER]", "] Nothing", " found."],
        ["[", "[", "N", "O", "_", "A", "N", "S", "W", "E", "R", "]", "]", "\n", "Nothing found."],
        ["  \n", "[[NO_ANSWER", "]]", " ", "Nothing found."],
        ["[[NO_ANSWER]]\n\nNothing", " found."],
    ],
)
def test_marker_split_across_tokens_never_leaks(deltas: list[str]) -> None:
    out, f = run(deltas)
    assert f.refused is True
    assert "".join(out) == "Nothing found."
    assert all("[" not in o and "]" not in o for o in out)
    assert "".join(deltas) == f.raw


def test_russian_refusal_after_marker() -> None:
    out, f = run(["[[NO_", "ANSWER]] ", REFUSAL_TEXTS["ru"]])
    assert f.refused is True
    assert "".join(out) == REFUSAL_TEXTS["ru"]


def test_normal_answer_streams_unchanged() -> None:
    deltas = ["Отпуск ", "можно ", "перенести [1]", "."]
    out, f = run(deltas)
    assert f.refused is False
    assert "".join(out) == "Отпуск можно перенести [1]."
    # Обычные токены не задерживаются дольше, чем нужно
    assert out[:2] == ["Отпуск ", "можно "]


def test_citation_brackets_are_not_held_forever() -> None:
    # «[» похоже на начало маркера, но «[1]» — нет: текст уходит пользователю без потерь
    out, f = run(["Yes [", "1", "] and [", "[", "2]]"])
    assert f.refused is False
    assert "".join(out) == "Yes [1] and [[2]]"


def test_partial_answer_is_not_refused() -> None:
    text = "The handbook covers leave [1], but the sources do not mention pensions."
    out, f = run([text[:20], text[20:]])
    assert f.refused is False
    assert "".join(out) == text


def test_marker_in_the_middle_is_stripped_but_not_a_refusal() -> None:
    out, f = run(["Leave is 25 days [1]. ", "[[NO_AN", "SWER]] Pension is not covered."])
    assert f.refused is False
    assert "".join(out) == "Leave is 25 days [1].  Pension is not covered."
    assert NO_ANSWER_MARKER not in "".join(out)


def test_unfinished_marker_prefix_is_flushed_at_the_end() -> None:
    out, f = run(["See section [[NO"])
    assert f.refused is False
    assert "".join(out) == "See section [[NO"


def test_marker_only_gives_empty_text() -> None:
    out, f = run(["[[NO_ANSWER", "]]", "  "])
    assert f.refused is True
    assert out == []


def test_strip_marker_on_whole_answer() -> None:
    assert strip_marker(f"  {NO_ANSWER_MARKER}\nNot found.") == ("Not found.", True)
    assert strip_marker("Answer [1].") == ("Answer [1].", False)


@pytest.mark.parametrize(
    ("question", "lang"),
    [
        ("Сколько дней отпуска можно перенести на следующий год?", "ru"),
        ("Как настроить Docker на сервере?", "ru"),
        ("What is the company pension contribution?", "en"),
        ("Can I carry over unused holiday?", "en"),
        ("Was ist die Rente?", "en"),
        ("", "en"),
        ("2 + 2?", "en"),
    ],
)
def test_detect_language(question: str, lang: str) -> None:
    assert detect_language(question) == lang


def test_no_chunks_refusal_follows_question_language() -> None:
    assert (
        refusal_text("Что такое пенсия?") == "В материалах агента я не нашёл ответа на этот вопрос."
    )
    assert refusal_text("What is the pension?") == (
        "I could not find an answer to this question in the agent's materials."
    )
    assert NO_ANSWER_MARKER not in refusal_text("anything")
