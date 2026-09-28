import pytest

from rag_agents.rag.chunking.structural import PARAGRAPH_SEP, StructuralChunker, context_header
from rag_agents.rag.parsing.base import Block, ParsedMeta, Section


class WordCounter:
    """Детерминированный счётчик для тестов: 1 слово = 1 токен."""

    def count(self, text: str) -> int:
        return len(text.split())

    def count_batch(self, texts: list[str]) -> list[int]:
        return [self.count(t) for t in texts]


NO_META = ParsedMeta(title=None, author=None)


def sec(path: list[str], *paras: str, page: int | None = None) -> Section:
    return Section(
        path=path,
        title=path[-1] if path else None,
        level=len(path),
        blocks=[Block(text=p, page=page) for p in paras],
    )


def sentence(n: int, word: str = "слово") -> str:
    return " ".join([word] * (n - 1) + [f"{word}."]).capitalize()


def chunker(**kw: int) -> StructuralChunker:
    params = {"target": 20, "max_tokens": 30, "min_tokens": 5, "overlap": 0} | kw
    return StructuralChunker(WordCounter(), **params)


def test_packs_paragraphs_up_to_target_without_crossing_sections() -> None:
    paras = [sentence(8) for _ in range(5)]  # 40 токенов в главе 1
    drafts = list(chunker().chunk(NO_META, [sec(["Гл. 1"], *paras), sec(["Гл. 2"], sentence(3))]))
    assert [(d.section_path, d.token_count) for d in drafts] == [
        (["Гл. 1"], 16),
        (["Гл. 1"], 16),
        (["Гл. 1"], 8),
        (["Гл. 2"], 3),  # короткая глава — свой чанк, даже меньше min
    ]
    assert [d.ord for d in drafts] == [0, 1, 2, 3]
    assert drafts[0].text == sentence(8) + PARAGRAPH_SEP + sentence(8)


def test_long_paragraph_is_split_by_sentences() -> None:
    para = " ".join(sentence(6) for _ in range(8))  # 48 токенов > max 30
    drafts = list(chunker().chunk(NO_META, [sec([], para)]))
    assert all(d.token_count <= 30 for d in drafts)
    assert all(d.text.endswith(".") for d in drafts)  # границы — по предложениям
    assert " ".join(d.text for d in drafts) == para


def test_overlap_repeats_last_sentences_of_previous_chunk() -> None:
    p1 = f"{sentence(10, 'альфа')} {sentence(4, 'бета')}"
    p2 = sentence(12, "гамма")
    drafts = list(chunker(overlap=5).chunk(NO_META, [sec([], p1, p2)]))
    assert len(drafts) == 2
    assert drafts[1].text.startswith(sentence(4, "бета"))
    assert drafts[1].text.endswith(p2)
    assert drafts[1].char_start < drafts[0].char_end  # перекрытие видно и по позициям


def test_small_tail_is_merged_into_previous_chunk() -> None:
    drafts = list(chunker(min_tokens=6).chunk(NO_META, [sec([], sentence(18), sentence(3))]))
    assert len(drafts) == 1
    assert drafts[0].token_count == 21


def test_word_split_for_giant_sentence() -> None:
    giant = " ".join(["слово"] * 70)  # без точек, 70 > max 30
    drafts = list(chunker().chunk(NO_META, [sec([], giant)]))
    assert [d.token_count for d in drafts] == [30, 30, 10]


def test_embed_text_has_context_header_and_normalized_text() -> None:
    meta = ParsedMeta(title="Исповедь", author="Толстой")
    [d] = list(chunker().chunk(meta, [sec(["Глава I"], "Мiръ и вѣра, ёж.")]))
    assert context_header(meta, ["Глава I"]) == "Толстой. Исповедь. Глава I"
    assert d.embed_text == "Толстой. Исповедь. Глава I\n\nМир и вера, еж."
    assert d.text == "Мiръ и вѣра, ёж."  # показ и цитаты — оригинал


def test_header_tokens_count_against_budget() -> None:
    meta = ParsedMeta(title="Книга", author="Автор")  # «Автор. Книга. Г» = 3 + 2 токена
    drafts = list(chunker().chunk(meta, [sec(["Г"], *[sentence(8) for _ in range(3)])]))
    # Без заголовка было бы [16, 8]; target 20 - 5 = 15 → по одному абзацу в чанке
    assert [d.token_count for d in drafts] == [8, 8, 8]


def test_offsets_and_pages() -> None:
    drafts = list(
        chunker(target=5, max_tokens=10, min_tokens=1).chunk(
            NO_META, [sec([], sentence(4), sentence(4), page=3)]
        )
    )
    assert (drafts[0].char_start, drafts[0].page_from, drafts[0].page_to) == (0, 3, 3)
    assert drafts[1].char_start == len(sentence(4)) + len(PARAGRAPH_SEP)


def test_invalid_params() -> None:
    with pytest.raises(ValueError, match="min"):
        StructuralChunker(WordCounter(), target=600, max_tokens=512)
    with pytest.raises(ValueError, match="overlap"):
        StructuralChunker(WordCounter(), target=100, max_tokens=200, min_tokens=10, overlap=100)
