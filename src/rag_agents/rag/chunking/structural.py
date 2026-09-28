"""Структурный чанкер под книги (ARCHITECTURE §7.4.2).

Внутри секции абзацы пакуются жадно до target токенов; длинный абзац режется по
предложениям (razdel), сверхдлинное предложение — по словам. Между соседними чанками
секции — overlap из последних предложений. Хвост меньше min приклеивается к предыдущему.
Чанк никогда не пересекает границу секции.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from razdel import sentenize

from rag_agents.domain.documents import ChunkDraft
from rag_agents.rag.chunking.tokenizer import TokenCounter
from rag_agents.rag.cleaning.orthography import norm_text
from rag_agents.rag.parsing.base import ParsedMeta, Section

PARAGRAPH_SEP = "\n\n"
_MIN_BUDGET = 8  # страховка от длинного заголовка: бюджет текста не уходит в ноль


def context_header(meta: ParsedMeta, section_path: list[str]) -> str:
    """«Автор. Книга. Часть / Глава» — contextual header перед текстом эмбеддинга."""
    parts = [p for p in (meta.author, meta.title, " / ".join(section_path) or None) if p]
    return ". ".join(parts)


@dataclass(frozen=True)
class _Unit:
    """Абзац или его часть (предложение, кусок предложения) с позицией в документе."""

    text: str
    tokens: int
    start: int
    end: int
    page: int | None
    new_para: bool  # перед юнитом — граница абзаца


@dataclass
class _Group:
    overlap: list[_Unit]
    body: list[_Unit]

    @property
    def body_tokens(self) -> int:
        return sum(u.tokens for u in self.body)

    @property
    def units(self) -> list[_Unit]:
        return [*self.overlap, *self.body]


class StructuralChunker:
    def __init__(
        self,
        counter: TokenCounter,
        *,
        target: int = 400,
        max_tokens: int = 512,
        min_tokens: int = 80,
        overlap: int = 60,
    ) -> None:
        if not 0 < min_tokens <= target <= max_tokens or overlap < 0 or overlap >= target:
            raise ValueError("need 0 < min <= target <= max and 0 <= overlap < target")
        self.counter = counter
        self.target = target
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap = overlap

    def chunk(self, meta: ParsedMeta, sections: Iterable[Section]) -> Iterator[ChunkDraft]:
        """Генератор: книга не держится в памяти целиком, чанки пишутся в PG батчами."""
        ord_ = 0
        offset = 0  # позиция в «очищенном тексте документа»: блоки через \n\n
        for section in sections:
            header = context_header(meta, section.path)
            header_tokens = self.counter.count(header) + 2 if header else 0
            target = max(self.target - header_tokens, min(self.target, _MIN_BUDGET))
            max_tokens = max(self.max_tokens - header_tokens, target)
            units, offset = self._units(section, max_tokens, offset)
            for group in self._merge_tail(self._pack(units, target, max_tokens), max_tokens):
                yield self._draft(ord_, section, header, group)
                ord_ += 1

    # --- разбиение на юниты ---

    def _units(self, section: Section, max_tokens: int, offset: int) -> tuple[list[_Unit], int]:
        texts = [b.text for b in section.blocks]
        counts = self.counter.count_batch(texts) if texts else []
        units: list[_Unit] = []
        for block, text, tokens in zip(section.blocks, texts, counts, strict=True):
            if tokens <= max_tokens:
                units.append(_Unit(text, tokens, offset, offset + len(text), block.page, True))
            else:
                units.extend(self._split_block(text, offset, block.page, max_tokens))
            offset += len(text) + len(PARAGRAPH_SEP)
        return units, offset

    def _split_block(
        self, text: str, offset: int, page: int | None, max_tokens: int
    ) -> list[_Unit]:
        sents = [(s.start, s.stop, s.text) for s in sentenize(text)]
        counts = self.counter.count_batch([t for _, _, t in sents])
        out: list[_Unit] = []
        for (start, stop, sent), tokens in zip(sents, counts, strict=True):
            if tokens <= max_tokens:
                out.append(_Unit(sent, tokens, offset + start, offset + stop, page, not out))
                continue
            # Предложение длиннее max (мусор PDF, списки без точек): режем по словам
            cur: list[str] = []
            pos = start
            for word in sent.split():
                if cur and self.counter.count(" ".join([*cur, word])) > max_tokens:
                    piece = " ".join(cur)
                    out.append(
                        _Unit(piece, self.counter.count(piece), offset + pos,
                              offset + pos + len(piece), page, not out)
                    )  # fmt: skip
                    pos += len(piece) + 1
                    cur = []
                cur.append(word)
            if cur:
                piece = " ".join(cur)
                out.append(
                    _Unit(piece, self.counter.count(piece), offset + pos,
                          offset + pos + len(piece), page, not out)
                )  # fmt: skip
        return out

    # --- упаковка ---

    def _pack(self, units: list[_Unit], target: int, max_tokens: int) -> list[_Group]:
        groups: list[_Group] = []
        overlap: list[_Unit] = []
        body: list[_Unit] = []
        tokens = 0
        for u in units:
            if body and tokens + u.tokens > target:
                groups.append(_Group(overlap, body))
                overlap = self._tail(body)
                body = []
                tokens = sum(x.tokens for x in overlap)
            if not body and tokens + u.tokens > max_tokens:
                overlap, tokens = [], 0  # overlap не помещается рядом с длинным юнитом
            body.append(u)
            tokens += u.tokens
        if body:
            groups.append(_Group(overlap, body))
        return groups

    def _tail(self, body: list[_Unit]) -> list[_Unit]:
        """Последние предложения чанка общим объёмом ≤ overlap токенов."""
        if self.overlap == 0:
            return []
        last = body[-1]
        sents = [(s.start, s.stop, s.text) for s in sentenize(last.text)]
        counts = self.counter.count_batch([t for _, _, t in sents]) if sents else []
        picked: list[_Unit] = []
        total = 0
        for (start, stop, text), tokens in zip(reversed(sents), reversed(counts), strict=True):
            if total + tokens > self.overlap:
                break
            picked.append(
                _Unit(text, tokens, last.start + start, last.start + stop, last.page, False)
            )
            total += tokens
        if len(picked) == len(sents):
            return []  # весь абзац в overlap — это уже не перекрытие, а дубль
        picked.reverse()
        return picked

    def _merge_tail(self, groups: list[_Group], max_tokens: int) -> list[_Group]:
        if len(groups) < 2 or groups[-1].body_tokens >= self.min_tokens:  # noqa: PLR2004
            return groups
        prev, last = groups[-2], groups[-1]
        if sum(u.tokens for u in prev.units) + last.body_tokens <= max_tokens:
            return [*groups[:-2], _Group(prev.overlap, [*prev.body, *last.body])]
        return groups

    # --- результат ---

    def _draft(self, ord_: int, section: Section, header: str, group: _Group) -> ChunkDraft:
        units = group.units
        parts: list[str] = []
        for i, u in enumerate(units):
            if i:
                parts.append(PARAGRAPH_SEP if u.new_para else " ")
            parts.append(u.text)
        text = "".join(parts)
        norm = norm_text(text)
        pages = [u.page for u in units if u.page is not None]
        return ChunkDraft(
            ord=ord_,
            section_path=section.path,
            chapter_title=section.title,
            text=text,
            embed_text=f"{header}\n\n{norm}" if header else norm,
            token_count=self.counter.count(text),
            char_start=units[0].start,
            char_end=units[-1].end,
            page_from=min(pages) if pages else None,
            page_to=max(pages) if pages else None,
        )
