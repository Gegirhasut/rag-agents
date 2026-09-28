"""FB2 и FB2.zip: потоковый iterparse с очисткой обработанных элементов (§7.2, §10.5)."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

from lxml import etree

from rag_agents.core.errors import PermanentError
from rag_agents.domain.enums import SourceFormat
from rag_agents.rag.parsing.base import ParsedMeta, Section
from rag_agents.rag.parsing.safety import LimitedReader, open_zip
from rag_agents.rag.parsing.sections import SectionBuilder

ZIP_MAGIC = b"PK\x03\x04"
NOTES_TITLE = "Примечания"
_BLOCK_TAGS = {"p", "v", "subtitle", "text-author", "td"}
_SKIP_TAGS = {"binary"}


def itertext(el: etree._Element) -> str:
    return "".join(t if isinstance(t, str) else t.decode() for t in el.itertext())


def _local(el: etree._Element) -> str:
    tag = el.tag
    return etree.QName(tag).localname if isinstance(tag, str) else ""


def render(el: etree._Element) -> str:
    """Текст элемента; ссылки на сноски `<a l:href="#n1">1</a>` → «[прим. 1]»."""
    parts = [el.text or ""]
    for child in el:
        if _local(child) == "a" and _is_note_link(child):
            label = itertext(child).strip().strip("[]") or "*"
            parts.append(f"[прим. {label}]")
        else:
            parts.append(render(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _is_note_link(a: etree._Element) -> bool:
    if a.get("type") == "note":
        return True
    return any(
        str(k).endswith("href") and str(v or "").startswith("#") for k, v in a.attrib.items()
    )


def _text(el: etree._Element | None) -> str | None:
    if el is None:
        return None
    s = " ".join(itertext(el).split())
    return s or None


@contextmanager
def _source(path: Path) -> Iterator[IO[bytes] | LimitedReader]:
    with path.open("rb") as f:
        is_zip = f.read(4) == ZIP_MAGIC
    if not is_zip:
        with path.open("rb") as f:
            yield f
        return
    with open_zip(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".fb2")]
        if not names:
            raise PermanentError("corrupted", "В архиве нет файла .fb2")
        with zf.open(names[0]) as raw:
            yield LimitedReader(raw)


def _iterparse(
    src: IO[bytes] | LimitedReader, events: tuple[str, ...]
) -> Iterator[tuple[str, etree._Element]]:
    try:
        yield from etree.iterparse(
            src,
            events=events,
            resolve_entities=False,
            no_network=True,
            load_dtd=False,
            huge_tree=False,
            recover=True,
        )
    except etree.XMLSyntaxError as e:
        raise PermanentError("corrupted", "Файл повреждён: некорректный XML") from e


class Fb2Parser:
    format = SourceFormat.FB2

    def parse(self, path: Path, filename: str) -> tuple[ParsedMeta, Iterator[Section]]:
        meta = self._meta(path)
        return meta, self._sections(path, meta)

    def _meta(self, path: Path) -> ParsedMeta:
        """Предварительный проход только по description: метаданные нужны до чанкинга."""
        with _source(path) as src:
            for _, el in _iterparse(src, ("end",)):
                name = _local(el)
                if name == "title-info":
                    return self._title_info(el)
                if name == "body":
                    break
        return ParsedMeta(title=None, author=None)

    @staticmethod
    def _title_info(el: etree._Element) -> ParsedMeta:
        def child(parent: etree._Element, name: str) -> etree._Element | None:
            return next((c for c in parent if _local(c) == name), None)

        author = None
        a = child(el, "author")
        if a is not None:
            names = [_text(child(a, n)) for n in ("first-name", "middle-name", "last-name")]
            author = " ".join(n for n in names if n) or _text(child(a, "nickname"))
        return ParsedMeta(
            title=_text(child(el, "book-title")),
            author=author,
            language=_text(child(el, "lang")),
            year=_text(child(el, "date")),
        )

    def _sections(self, path: Path, meta: ParsedMeta) -> Iterator[Section]:  # noqa: PLR0912, PLR0915  конечный автомат по событиям iterparse читается подряд
        builder = SectionBuilder()
        depth = 0  # вложенность section в основном body
        in_title = 0
        in_notes = False
        in_body = False
        note_title: str | None = None
        note_parts: list[str] = []
        native = 0

        with _source(path) as src:
            for event, el in _iterparse(src, ("start", "end")):
                name = _local(el)
                if event == "start":
                    if name == "body":
                        in_body = True
                        in_notes = el.get("name") in ("notes", "comments")
                        if in_notes:
                            yield from builder.heading(1, NOTES_TITLE)
                    elif name == "section" and in_body and not in_notes:
                        yield from builder.flush()
                        depth += 1
                    elif name == "title":
                        in_title += 1
                    continue

                # --- end ---
                if name in _SKIP_TAGS:
                    el.clear()
                elif not in_body:
                    continue
                elif name == "title":
                    in_title -= 1
                    title = " ".join(
                        filter(None, (_text(p) for p in el.iter() if _local(p) == "p"))
                    ) or _text(el)
                    parent = el.getparent()
                    if in_notes:
                        note_title = title
                    elif parent is not None and _local(parent) == "section" and title:
                        yield from builder.heading(depth, title)
                        native += 1
                    el.clear()
                elif name in _BLOCK_TAGS and not in_title:
                    text = render(el)
                    if in_notes:
                        note_parts.append(text)
                    else:
                        builder.add(text)
                    el.clear()
                elif name == "section":
                    if in_notes:
                        if note_parts:
                            label = f"[прим. {note_title}] " if note_title else ""
                            builder.add(label + " ".join(note_parts))
                        note_title, note_parts = None, []
                    else:
                        yield from builder.flush()
                        depth -= 1
                        builder.stack.truncate(depth)
                    _free(el)
                elif name == "body":
                    yield from builder.flush()
                    builder.stack.truncate(0)
                    in_body = in_notes = False
                    depth = 0
                    _free(el)
        yield from builder.flush()
        meta.toc_source = "native" if native else "none"


def _free(el: etree._Element) -> None:
    """Классический приём iterparse: чистим элемент и уже обработанных соседей."""
    el.clear()
    parent = el.getparent()
    if parent is not None:
        while el.getprevious() is not None:
            del parent[0]
