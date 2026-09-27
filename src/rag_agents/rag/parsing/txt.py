import codecs
import re
import unicodedata
from collections.abc import Iterator
from pathlib import Path

from charset_normalizer import from_bytes

from rag_agents.core.errors import PermanentError
from rag_agents.domain.enums import SourceFormat
from rag_agents.rag.parsing.base import Block, ParsedMeta, Section

_SAMPLE_BYTES = 64 * 1024
_MIN_CYRILLIC_SHARE = 0.3
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_LETTERS = re.compile(r"[^\W\d_]")
_CONTROL = re.compile("[\u00ad\u200b\u200c\u200d\ufeff]")  # soft hyphen, zero-width, BOM
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
    stem = Path(filename).stem.replace("_", " ").strip()
    m = _AUTHOR_TITLE.match(stem)
    if m:
        return m.group("author").strip(), m.group("title").strip()
    return None, stem


def split_paragraphs(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\n\n" in text:
        raw = re.split(r"\n\s*\n", text)
        # Внутри абзаца переносы строк — артефакт вёрстки
        paras = [re.sub(r"\s*\n\s*", " ", p) for p in raw]
    else:
        paras = text.split("\n")
    return [p for p in (re.sub(r"[ \t ]+", " ", p).strip() for p in paras) if p]


class TxtParser:
    """Итерация 1: весь документ — одна секция. Эвристика глав появится в итерации 3."""

    format = SourceFormat.TXT

    def parse(self, path: Path, filename: str) -> tuple[ParsedMeta, Iterator[Section]]:
        data = path.read_bytes()
        if not data.strip():
            raise PermanentError("empty_document", "Файл пустой")
        encoding = detect_encoding(data)
        text = unicodedata.normalize("NFC", data.decode(encoding, errors="replace"))
        text = _CONTROL.sub("", text)
        paragraphs = split_paragraphs(text)
        if not paragraphs:
            raise PermanentError("empty_document", "В файле нет текста")
        author, title = meta_from_filename(filename)
        meta = ParsedMeta(title=title, author=author, encoding=encoding)
        section = Section(path=[], title=None, level=0, blocks=[Block(text=p) for p in paragraphs])
        return meta, iter([section])
