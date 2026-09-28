"""Формат по расширению + проверка magic bytes (ARCHITECTURE §7.1, §16)."""

from pathlib import PurePath

from rag_agents.domain.enums import SourceFormat

EXTENSIONS: dict[str, SourceFormat] = {
    ".txt": SourceFormat.TXT,
    ".fb2": SourceFormat.FB2,
    ".fb2.zip": SourceFormat.FB2,
    ".epub": SourceFormat.EPUB,
    ".pdf": SourceFormat.PDF,
    ".docx": SourceFormat.DOCX,
}
_ZIP = b"PK\x03\x04"


class UnsupportedFormatError(Exception):
    pass


def format_by_name(filename: str) -> SourceFormat:
    name = PurePath(filename).name.lower()
    if name.endswith(".fb2.zip"):
        return SourceFormat.FB2
    suffix = PurePath(name).suffix
    if suffix in EXTENSIONS:
        return EXTENSIONS[suffix]
    raise UnsupportedFormatError(
        f"Неподдерживаемый формат: {suffix or 'без расширения'}. "
        "Можно: .txt, .fb2, .fb2.zip, .epub, .pdf, .docx"
    )


def magic_matches(fmt: SourceFormat, head: bytes) -> bool:
    """Содержимое соответствует расширению: .pdf, переименованный из .exe, не пройдёт."""
    match fmt:
        case SourceFormat.PDF:
            return b"%PDF-" in head[:1024]
        case SourceFormat.EPUB | SourceFormat.DOCX:
            return head.startswith(_ZIP)
        case SourceFormat.FB2:
            if head.startswith(_ZIP):
                return True
            text = head.lstrip(b"\xef\xbb\xbf \t\r\n")
            return text.startswith((b"<?xml", b"<FictionBook"))
        case SourceFormat.TXT:
            # Бинарные файлы почти всегда содержат NUL в первых килобайтах
            return b"\x00" not in head[:8192] and not head.startswith((_ZIP, b"%PDF-"))
