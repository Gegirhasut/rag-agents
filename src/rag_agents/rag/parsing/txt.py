import codecs
import re
from collections.abc import Iterator
from pathlib import Path

from charset_normalizer import from_bytes

from rag_agents.core.errors import PermanentError
from rag_agents.domain.enums import SourceFormat
from rag_agents.rag.parsing.base import ParsedMeta, Section
from rag_agents.rag.parsing.headings import heading_level
from rag_agents.rag.parsing.sections import SectionBuilder

_SAMPLE_BYTES = 64 * 1024
_READ_BYTES = 1024 * 1024
_MIN_CYRILLIC_SHARE = 0.3
# Абзацы разделены пустыми строками, если их хотя бы 5 % строк сэмпла; иначе абзац = строка
_BLANK_LINE_SHARE = 0.05
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_LETTERS = re.compile(r"[^\W\d_]")
# «Толстой - Исповедь.txt», «Толстой — Исповедь.txt»
_AUTHOR_TITLE = re.compile(r"^\s*(?P<author>[^-—–]{2,80}?)\s+[-—–]\s+(?P<title>.{1,200})$")


def detect_encoding(data: bytes) -> str:
    """UTF-8 пробуем строго; иначе charset-normalizer c проверкой, что вышла кириллица.

    cp1251 и koi8-r в русских txt встречаются часто, а ASCII-эвристики их путают.
    """
    sample = data[:_SAMPLE_BYTES]
    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        # Инкрементальный декодер не падает на многобайтном символе, разрезанном концом сэмпла
        codecs.getincrementaldecoder("utf-8")().decode(sample, final=len(data) <= _SAMPLE_BYTES)
    except UnicodeDecodeError:
        pass
    else:
        return "utf-8"

    candidates = [
        m.encoding for m in from_bytes(sample, cp_isolation=["cp1251", "koi8_r", "cp866"])
    ]
    candidates += ["cp1251", "koi8_r"]
    for enc in candidates:
        text = sample.decode(enc, errors="replace")
        letters = _LETTERS.findall(text)
        cyr = _CYRILLIC.findall(text)
        if letters and len(cyr) / len(letters) >= _MIN_CYRILLIC_SHARE:
            return enc
    raise PermanentError("unknown_encoding", "Не удалось определить кодировку текста")


def meta_from_filename(filename: str) -> tuple[str | None, str]:
    name = filename
    for suffix in (".zip", ".txt", ".fb2", ".epub", ".pdf", ".docx"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    stem = Path(name).name.replace("_", " ").strip()
    m = _AUTHOR_TITLE.match(stem)
    if m:
        return m.group("author").strip(), m.group("title").strip()
    return None, stem


def _iter_lines(path: Path, encoding: str) -> Iterator[str]:
    """Потоковое декодирование: память не зависит от размера файла."""
    decoder = codecs.getincrementaldecoder(encoding)(errors="replace")
    tail = ""
    with path.open("rb") as f:
        while chunk := f.read(_READ_BYTES):
            text = tail + decoder.decode(chunk)
            lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            tail = lines.pop()
            yield from lines
    rest = tail + decoder.decode(b"", final=True)
    if rest:
        yield rest


def _blank_line_mode(sample: str) -> bool:
    lines = sample.splitlines()
    blanks = sum(1 for x in lines if not x.strip())
    return bool(lines) and blanks / len(lines) >= _BLANK_LINE_SHARE


class TxtParser:
    """Кодировка по сэмплу, потоковое чтение, главы — эвристикой заголовков (§7.4.1)."""

    format = SourceFormat.TXT

    def parse(self, path: Path, filename: str) -> tuple[ParsedMeta, Iterator[Section]]:
        with path.open("rb") as f:
            head = f.read(_SAMPLE_BYTES + 4)
        if not head.strip():
            raise PermanentError("empty_document", "Файл пустой")
        encoding = detect_encoding(head)
        author, title = meta_from_filename(filename)
        meta = ParsedMeta(title=title, author=author, encoding=encoding)
        blank_mode = _blank_line_mode(head.decode(encoding, errors="replace"))
        return meta, self._sections(path, encoding, meta, blank_mode=blank_mode)

    def _sections(
        self, path: Path, encoding: str, meta: ParsedMeta, *, blank_mode: bool
    ) -> Iterator[Section]:
        builder = SectionBuilder()
        para: list[str] = []

        def close() -> Iterator[Section]:
            if not para:
                return
            text = " ".join(x.strip() for x in para)
            # Заголовок — одиночная короткая строка, окружённая пустыми строками
            level = heading_level(text) if len(para) == 1 else None
            if level is not None:
                yield from builder.heading(level, text)
            else:
                builder.add(text)
            para.clear()

        for line in _iter_lines(path, encoding):
            if not line.strip():
                yield from close()
            elif blank_mode:
                para.append(line)
            else:
                para.append(line)
                yield from close()
        yield from close()
        yield from builder.flush()
        meta.toc_source = "heuristic" if builder.headings else "none"
