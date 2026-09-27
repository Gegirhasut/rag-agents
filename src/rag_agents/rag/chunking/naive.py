import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from rag_agents.domain.documents import ChunkDraft
from rag_agents.rag.chunking.tokenizer import TokenCounter
from rag_agents.rag.parsing.base import ParsedMeta, Section

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[«\"(A-ZА-ЯЁ—-])")
PARAGRAPH_SEP = "\n\n"


def normalize_for_index(text: str) -> str:
    """Текст для эмбеддинга и (позже) BM25: ё→е. Показ и цитаты идут по исходному тексту."""
    return unicodedata.normalize("NFC", text).replace("ё", "е").replace("Ё", "Е")


def context_header(meta: ParsedMeta, section_path: list[str]) -> str:
    parts = [p for p in (meta.author, meta.title, " / ".join(section_path) or None) if p]
    return ". ".join(parts)


@dataclass
class _Piece:
    text: str
    tokens: int


class NaiveChunker:
    """Итерация 1: жадная упаковка абзацев до target токенов, без overlap.

    Слишком длинный абзац режется по предложениям, слишком длинное предложение — по словам.
    Структурный чанкер с overlap и razdel — итерация 3 (ARCHITECTURE §7.4).
    """

    def __init__(self, counter: TokenCounter, target: int = 400, max_tokens: int = 512) -> None:
        if not 0 < target <= max_tokens:
            raise ValueError("need 0 < target <= max_tokens")
        self.counter = counter
        self.target = target
        self.max_tokens = max_tokens

    def chunk(self, meta: ParsedMeta, sections: Iterable[Section]) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        offset = 0  # позиция в «очищенном тексте документа»: абзацы, склеенные через \n\n
        for section in sections:
            header = context_header(meta, section.path)
            header_tokens = self.counter.count(header) + 2 if header else 0
            budget_target = max(self.target - header_tokens, 1)
            budget_max = max(self.max_tokens - header_tokens, 1)
            pieces = self._pieces([b.text for b in section.blocks], budget_max)
            buf: list[_Piece] = []
            buf_tokens = 0
            for piece in pieces:
                if buf and buf_tokens + piece.tokens > budget_target:
                    offset = self._emit(drafts, buf, section, header, offset)
                    buf, buf_tokens = [], 0
                buf.append(piece)
                buf_tokens += piece.tokens
            if buf:
                offset = self._emit(drafts, buf, section, header, offset)
        return drafts

    def _pieces(self, paragraphs: list[str], budget_max: int) -> list[_Piece]:
        counts = self.counter.count_batch(paragraphs)
        out: list[_Piece] = []
        for text, tokens in zip(paragraphs, counts, strict=True):
            if tokens <= budget_max:
                out.append(_Piece(text, tokens))
            else:
                out.extend(self._split_long(text, budget_max))
        return out

    def _split_long(self, text: str, budget_max: int) -> list[_Piece]:
        units = _SENTENCE_END.split(text)
        out: list[_Piece] = []
        cur: list[str] = []
        for unit in units:
            unit_tokens = self.counter.count(unit)
            if unit_tokens > budget_max:
                if cur:
                    out.append(self._piece(" ".join(cur)))
                    cur = []
                out.extend(self._split_words(unit, budget_max))
                continue
            candidate = " ".join([*cur, unit])
            if cur and self.counter.count(candidate) > budget_max:
                out.append(self._piece(" ".join(cur)))
                cur = [unit]
            else:
                cur.append(unit)
        if cur:
            out.append(self._piece(" ".join(cur)))
        return out

    def _split_words(self, text: str, budget_max: int) -> list[_Piece]:
        out: list[_Piece] = []
        cur: list[str] = []
        for word in text.split():
            if cur and self.counter.count(" ".join([*cur, word])) > budget_max:
                out.append(self._piece(" ".join(cur)))
                cur = []
            cur.append(word)
        if cur:
            out.append(self._piece(" ".join(cur)))
        return out

    def _piece(self, text: str) -> _Piece:
        return _Piece(text, self.counter.count(text))

    def _emit(
        self,
        drafts: list[ChunkDraft],
        buf: list[_Piece],
        section: Section,
        header: str,
        offset: int,
    ) -> int:
        text = PARAGRAPH_SEP.join(p.text for p in buf)
        norm = normalize_for_index(text)
        embed_text = f"{header}\n\n{norm}" if header else norm
        drafts.append(
            ChunkDraft(
                ord=len(drafts),
                section_path=section.path,
                chapter_title=section.title,
                text=text,
                embed_text=embed_text,
                token_count=sum(p.tokens for p in buf),
                char_start=offset,
                char_end=offset + len(text),
            )
        )
        return offset + len(text) + len(PARAGRAPH_SEP)
