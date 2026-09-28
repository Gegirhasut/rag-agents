"""Защита парсеров от враждебных файлов (ARCHITECTURE §16): zip-bomb и XXE."""

import zipfile
from pathlib import Path
from typing import IO

from lxml import etree

from rag_agents.core.errors import PermanentError

MAX_UNZIPPED_BYTES = 300 * 1024 * 1024
MAX_ZIP_MEMBERS = 10_000
# Коэффициент сжатия текста редко выше 20; сотни — признак zip-bomb
MAX_COMPRESSION_RATIO = 200


def open_zip(path: Path) -> zipfile.ZipFile:
    """Открывает архив и проверяет заявленные размеры до распаковки."""
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as e:
        raise PermanentError("corrupted", "Файл повреждён: не удалось открыть архив") from e
    infos = zf.infolist()
    total = sum(i.file_size for i in infos)
    packed = sum(i.compress_size for i in infos) or 1
    if (
        len(infos) > MAX_ZIP_MEMBERS
        or total > MAX_UNZIPPED_BYTES
        or total / packed > MAX_COMPRESSION_RATIO
    ):
        zf.close()
        raise PermanentError("zip_bomb", "Архив распаковывается в слишком большой объём")
    return zf


class LimitedReader:
    """Файловый объект поверх распаковки: заголовок zip может врать о размере, поэтому
    считаем реально прочитанные байты."""

    def __init__(self, raw: IO[bytes], limit: int = MAX_UNZIPPED_BYTES) -> None:
        self.raw = raw
        self.limit = limit
        self.read_bytes = 0

    def read(self, n: int = -1) -> bytes:
        data = self.raw.read(n)
        self.read_bytes += len(data)
        if self.read_bytes > self.limit:
            raise PermanentError("zip_bomb", "Архив распаковывается в слишком большой объём")
        return data


def read_member(zf: zipfile.ZipFile, name: str) -> bytes:
    with zf.open(name) as raw:
        reader = LimitedReader(raw)
        return reader.read()


def xml_parser() -> etree.XMLParser:
    """Без внешних сущностей, DTD и сети: XXE и «billion laughs» не проходят."""
    return etree.XMLParser(
        resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False, recover=True
    )


def parse_xml(data: bytes) -> etree._Element:
    try:
        root = etree.fromstring(data, parser=xml_parser())
    except etree.XMLSyntaxError as e:
        raise PermanentError("corrupted", "Файл повреждён: некорректный XML") from e
    if root is None:
        raise PermanentError("corrupted", "Файл повреждён: пустой XML")
    return root
