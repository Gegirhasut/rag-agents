import re

import nh3
from markdown_it import MarkdownIt

_md = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable("table")
_CITE = re.compile(r"\[(\d{1,2})\]")
_ALLOWED_TAGS = {
    "p", "br", "strong", "em", "ul", "ol", "li", "blockquote", "code", "pre",
    "h2", "h3", "h4", "table", "thead", "tbody", "tr", "th", "td", "a", "sup", "hr",
}  # fmt: skip
_ALLOWED_ATTRS = {"a": {"href", "class", "data-cite"}}


def render_answer(markdown: str, anchor_prefix: str, citation_count: int) -> str:
    """Markdown ответа → безопасный HTML; [n] → якоря на карточки источников."""
    html = _md.render(markdown)

    def link(m: re.Match[str]) -> str:
        n = int(m.group(1))
        if 1 <= n <= citation_count:
            return f'<a href="#{anchor_prefix}-{n}" class="cite" data-cite="{n}">[{n}]</a>'
        return m.group(0)

    html = _CITE.sub(link, html)
    return nh3.clean(html, tags=_ALLOWED_TAGS, attributes=_ALLOWED_ATTRS, link_rel=None)
