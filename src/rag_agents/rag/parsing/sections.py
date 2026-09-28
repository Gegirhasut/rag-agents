from collections.abc import Iterator

from rag_agents.rag.cleaning.normalize import clean_text, is_junk
from rag_agents.rag.parsing.base import Block, Section
from rag_agents.rag.parsing.headings import HeadingStack


class SectionBuilder:
    """Копит блоки текущей секции; заголовок закрывает её и открывает новую.

    Общий для всех парсеров: очистка блока (§7.3) и отсев мусора — здесь, в одном месте.
    """

    def __init__(self) -> None:
        self.stack = HeadingStack()
        self.blocks: list[Block] = []
        self.headings = 0

    def add(self, text: str, page: int | None = None) -> None:
        text = clean_text(text)
        if text and not is_junk(text):
            self.blocks.append(Block(text=text, page=page))

    def heading(self, level: int, title: str) -> Iterator[Section]:
        """Отдаёт накопленную секцию (если в ней есть текст) и переключает путь."""
        yield from self.flush()
        title = " ".join(clean_text(title).split())
        if title:
            self.stack.push(level, title)
            self.headings += 1

    def blocks_on_page(self, page: int) -> bool:
        return any(b.page == page for b in self.blocks)

    def flush(self) -> Iterator[Section]:
        if self.blocks:
            path = self.stack.path
            yield Section(
                path=path,
                title=path[-1] if path else None,
                level=self.stack.level,
                blocks=self.blocks,
            )
        self.blocks = []
