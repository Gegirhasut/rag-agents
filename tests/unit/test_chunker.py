import pytest

from rag_agents.rag.chunking.naive import NaiveChunker, normalize_for_index
from rag_agents.rag.parsing.base import Block, ParsedMeta, Section


class WordCounter:
    """Детерминированный счётчик для тестов: 1 слово = 1 токен."""

    def count(self, text: str) -> int:
        return len(text.split())

    def count_batch(self, texts: list[str]) -> list[int]:
        return [self.count(t) for t in texts]


META = ParsedMeta(title="Исповедь", author="Толстой")


def section(*paragraphs: str) -> Section:
    return Section(path=[], title=None, level=0, blocks=[Block(text=p) for p in paragraphs])


def words(n: int, prefix: str = "слово") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_packs_paragraphs_up_to_target() -> None:
    # Заголовок «Толстой. Исповедь» = 2 токена + 2 → бюджет target 10 - 4 = 6
    chunker = NaiveChunker(WordCounter(), target=10, max_tokens=20)
    drafts = chunker.chunk(META, [section(words(3, "а"), words(3, "б"), words(3, "в"))])

    assert [d.token_count for d in drafts] == [6, 3]
    assert drafts[0].text == f"{words(3, 'а')}\n\n{words(3, 'б')}"
    assert [d.ord for d in drafts] == [0, 1]


def test_long_paragraph_is_split_by_sentences_then_words() -> None:
    chunker = NaiveChunker(WordCounter(), target=8, max_tokens=10)
    long_sentence = words(15, "x") + "."
    para = f"Короткое первое предложение. {long_sentence} Хвост."
    drafts = chunker.chunk(META, [section(para)])

    budget_max = 10 - 4
    assert all(d.token_count <= budget_max for d in drafts)
    joined = " ".join(d.text for d in drafts)
    assert joined.split() == para.split()  # ничего не потеряно и не переставлено


def test_char_offsets_are_contiguous_in_cleaned_text() -> None:
    chunker = NaiveChunker(WordCounter(), target=6, max_tokens=12)
    paras = [words(2, "a"), words(2, "b"), words(2, "c")]
    drafts = chunker.chunk(ParsedMeta(title=None, author=None), [section(*paras)])
    document = "\n\n".join(paras)
    for d in drafts:
        assert document[d.char_start : d.char_end] == d.text


def test_embed_text_has_context_header_and_normalized_yo() -> None:
    chunker = NaiveChunker(WordCounter(), target=50, max_tokens=60)
    [d] = chunker.chunk(META, [section("Я был крещён.")])
    assert d.text == "Я был крещён."
    assert d.embed_text == "Толстой. Исповедь\n\nЯ был крещен."


def test_normalize_for_index() -> None:
    assert normalize_for_index("Ёлка ещё") == "Елка еще"


def test_rejects_bad_limits() -> None:
    with pytest.raises(ValueError, match="target"):
        NaiveChunker(WordCounter(), target=600, max_tokens=512)
