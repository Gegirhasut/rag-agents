"""EPUB без ebooklib (AGPL): OPF и оглавление читаем сами (lxml), XHTML — selectolax."""

import posixpath
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urldefrag

from lxml import etree
from selectolax.parser import HTMLParser, Node

from rag_agents.core.errors import PermanentError
from rag_agents.domain.enums import SourceFormat
from rag_agents.rag.parsing.base import ParsedMeta, Section
from rag_agents.rag.parsing.fb2 import itertext
from rag_agents.rag.parsing.safety import open_zip, parse_xml, read_member
from rag_agents.rag.parsing.sections import SectionBuilder

_BLOCKS = frozenset(
    {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "dd", "dt", "pre"}
)
_BLOCK_SELECTOR = ", ".join(sorted(_BLOCKS))
_HEADINGS = {"h1", "h2", "h3"}
_SKIP_TYPES = {"cover", "toc", "nav", "titlepage", "copyright-page"}
_XHTML_TYPES = {"application/xhtml+xml", "text/html"}


@dataclass(frozen=True)
class TocEntry:
    path: list[str]  # с учётом вложенности оглавления


def _local(el: etree._Element) -> str:
    return etree.QName(el.tag).localname if isinstance(el.tag, str) else ""


def _first(root: etree._Element, name: str) -> str | None:
    for el in root.iter():
        if _local(el) == name:
            text = " ".join(itertext(el).split())
            if text:
                return text
    return None


class EpubParser:
    format = SourceFormat.EPUB

    def parse(self, path: Path, filename: str) -> tuple[ParsedMeta, Iterator[Section]]:
        zf = open_zip(path)
        try:
            opf_path = self._opf_path(zf)
            opf = parse_xml(read_member(zf, opf_path))
        except (KeyError, PermanentError) as e:
            zf.close()
            if isinstance(e, PermanentError):
                raise
            raise PermanentError("corrupted", "Файл повреждён: не найден OPF") from e
        meta = ParsedMeta(
            title=_first(opf, "title"),
            author=_first(opf, "creator"),
            language=_first(opf, "language"),
            year=_first(opf, "date"),
        )
        return meta, self._sections(zf, opf_path, opf, meta)

    @staticmethod
    def _opf_path(zf: zipfile.ZipFile) -> str:
        container = parse_xml(read_member(zf, "META-INF/container.xml"))
        for el in container.iter():
            if _local(el) == "rootfile" and el.get("full-path"):
                return str(el.get("full-path"))
        raise PermanentError("corrupted", "Файл повреждён: нет rootfile в container.xml")

    def _sections(
        self, zf: zipfile.ZipFile, opf_path: str, opf: etree._Element, meta: ParsedMeta
    ) -> Iterator[Section]:
        base = posixpath.dirname(opf_path)
        manifest: dict[str, tuple[str, str, str]] = {}  # id → (href, media-type, properties)
        spine: list[str] = []
        toc_id = None
        guide_skip: set[str] = set()
        for el in opf.iter():
            name = _local(el)
            if name == "item":
                href = posixpath.normpath(posixpath.join(base, unquote(el.get("href", ""))))
                manifest[el.get("id", "")] = (
                    href,
                    el.get("media-type", ""),
                    el.get("properties", ""),
                )
            elif name == "itemref" and el.get("linear") != "no":
                spine.append(el.get("idref", ""))
            elif name == "spine":
                toc_id = el.get("toc")
            elif name == "reference" and el.get("type") in _SKIP_TYPES:
                guide_skip.add(
                    posixpath.normpath(posixpath.join(base, urldefrag(el.get("href", ""))[0]))
                )

        toc = self._toc(zf, manifest, toc_id)
        meta.toc_source = "native" if toc else "heuristic"
        builder = SectionBuilder()
        try:
            for idref in spine:
                href, media, props = manifest.get(idref, ("", "", ""))
                if media not in _XHTML_TYPES or "nav" in props.split() or href in guide_skip:
                    continue
                try:
                    html = read_member(zf, href)
                except KeyError:
                    continue
                yield from self._document(builder, html, toc.get(href))
            yield from builder.flush()
        finally:
            zf.close()

    @staticmethod
    def _document(
        builder: SectionBuilder, html: bytes, entry: TocEntry | None
    ) -> Iterator[Section]:
        tree = HTMLParser(html)
        body = tree.body or tree.root
        if body is None:
            return
        # traverse — порядок документа (css() отдаёт узлы в порядке селекторов)
        nodes = [n for n in body.traverse() if n.tag in _BLOCKS and not _has_block_child(n)]
        first_heading = next((n for n in nodes if n.tag in _HEADINGS), None)
        # Секция прошлого файла закрывается до смены пути
        yield from builder.flush()
        if entry is not None:
            path = entry.path
            builder.stack.truncate(0)
            for level, title in enumerate(path[:-1], start=1):
                builder.stack.push(level, title)
            yield from builder.heading(len(path), path[-1])
        elif first_heading is not None:
            title = first_heading.text(separator=" ", strip=True)
            if title:
                yield from builder.heading(int(first_heading.tag[1]), title)
        heading_text = first_heading.text(separator=" ", strip=True) if first_heading else None
        for n in nodes:
            text = n.text(separator=" ", strip=True)
            if n is first_heading or (n.tag in _HEADINGS and text == heading_text):
                continue
            builder.add(text)

    def _toc(
        self,
        zf: zipfile.ZipFile,
        manifest: dict[str, tuple[str, str, str]],
        toc_id: str | None,
    ) -> dict[str, TocEntry]:
        """Файл главы → путь в оглавлении. EPUB 3 — nav-документ, EPUB 2 — NCX."""
        nav = next((m for m in manifest.values() if "nav" in m[2].split()), None)
        try:
            if nav is not None:
                return self._nav_toc(read_member(zf, nav[0]), posixpath.dirname(nav[0]))
            ncx = manifest.get(toc_id or "") or next(
                (m for m in manifest.values() if m[1] == "application/x-dtbncx+xml"), None
            )
            if ncx is not None:
                return self._ncx_toc(read_member(zf, ncx[0]), posixpath.dirname(ncx[0]))
        except (KeyError, PermanentError):
            return {}
        return {}

    @staticmethod
    def _nav_toc(html: bytes, base: str) -> dict[str, TocEntry]:
        tree = HTMLParser(html)
        navs = tree.css("nav")
        nav = next(
            (n for n in navs if "toc" in (n.attributes.get("epub:type") or "")),
            navs[0] if navs else None,
        )
        out: dict[str, TocEntry] = {}
        if nav is None:
            return out

        def walk(ol: Node, parents: list[str]) -> None:
            for li in ol.iter():
                if li.tag != "li":
                    continue
                a = li.css_first("a")
                title = a.text(separator=" ", strip=True) if a else ""
                path = [*parents, title] if title else parents
                if a is not None and title:
                    href = urldefrag(unquote(a.attributes.get("href") or ""))[0]
                    key = posixpath.normpath(posixpath.join(base, href))
                    out.setdefault(key, TocEntry(path))
                sub = li.css_first("ol")
                if sub is not None:
                    walk(sub, path)

        root_ol = nav.css_first("ol")
        if root_ol is not None:
            walk(root_ol, [])
        return out

    @staticmethod
    def _ncx_toc(data: bytes, base: str) -> dict[str, TocEntry]:
        root = parse_xml(data)
        out: dict[str, TocEntry] = {}

        def walk(el: etree._Element, parents: list[str]) -> None:
            for point in el:
                if _local(point) != "navPoint":
                    continue
                title = _first(point, "text") or ""
                content = next((c for c in point if _local(c) == "content"), None)
                path = [*parents, title] if title else parents
                if content is not None and title:
                    href = urldefrag(unquote(content.get("src", "")))[0]
                    out.setdefault(posixpath.normpath(posixpath.join(base, href)), TocEntry(path))
                walk(point, path)

        nav_map = next((e for e in root.iter() if _local(e) == "navMap"), None)
        if nav_map is not None:
            walk(nav_map, [])
        return out


def _has_block_child(node: Node) -> bool:
    """li с вложенными p и blockquote с абзацами: берём внутренние, а не контейнер."""
    # traverse() идёт дальше поддерева, поэтому ищем через css() внутри узла; обёртки
    # selectolax каждый раз новые — сам узел отсекаем по mem_id
    return any(child.mem_id != node.mem_id for child in node.css(_BLOCK_SELECTOR))
