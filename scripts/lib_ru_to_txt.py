"""HTML-страница произведения с az.lib.ru (cp1251) → plain txt в cp1251.

Подготовка тестового корпуса:
    curl ... | python3 scripts/lib_ru_to_txt.py > book.txt
"""

import html
import re
import sys


def convert(raw: bytes) -> str:
    page = raw.decode("cp1251", errors="replace")
    body = page.split("<!--Section Begins-->", 1)[-1]
    body = body.split("<!--Section Ends-->", 1)[0]
    body = re.sub(r"(?i)<br\s*/?>|</?(p|dd|div|h\d|center)[^>]*>", "\n", body)
    body = re.sub(r"<[^>]+>", "", body)
    body = html.unescape(body).replace("\r", "")
    lines = (line.strip() for line in body.split("\n"))
    return "\n\n".join(line for line in lines if line) + "\n"


if __name__ == "__main__":
    sys.stdout.buffer.write(convert(sys.stdin.buffer.read()).encode("cp1251", errors="replace"))
