"""Отказ «не нашёл»: служебный маркер от LLM и текст отказа на языке вопроса.

LLM начинает отказ с маркера NO_ANSWER_MARKER, а дальше пишет одну фразу на языке вопроса.
По маркеру ставится refused=True; пользователю (SSE-стрим, answer_md, история чата) маркер
не попадает: MarkerFilter вырезает его из потока токенов, даже если он разбит на части.
"""

import re

NO_ANSWER_MARKER = "[[NO_ANSWER]]"

# Детерминированный отказ без вызова LLM (ничего не найдено); ru — исходная фраза
REFUSAL_TEXTS: dict[str, str] = {
    "ru": "В материалах агента я не нашёл ответа на этот вопрос.",
    "en": "I could not find an answer to this question in the agent's materials.",
}

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_LATIN = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """Грубая эвристика по письменности: кириллицы больше, чем латиницы → ru, иначе en."""
    cyr = len(_CYRILLIC.findall(text))
    return "ru" if cyr and cyr >= len(_LATIN.findall(text)) else "en"


def refusal_text(question: str) -> str:
    return REFUSAL_TEXTS[detect_language(question)]


def _held_prefix_len(text: str, marker: str) -> int:
    """Длина самого длинного хвоста text, который может оказаться началом marker."""
    for n in range(min(len(text), len(marker) - 1), 0, -1):
        if marker.startswith(text[-n:]):
            return n
    return 0


class MarkerFilter:
    """Потоковый фильтр токенов LLM: вырезает NO_ANSWER_MARKER и определяет отказ.

    feed() возвращает текст, который можно отдать пользователю сейчас; хвост, похожий на
    начало маркера, придерживается до следующего токена. flush() отдаёт остаток в конце стрима.
    refused=True, только если ответ начинается с маркера (ведущие пробелы не считаются);
    маркер в середине ответа просто вырезается, отказом это не считается.
    """

    def __init__(self, marker: str = NO_ANSWER_MARKER) -> None:
        self.marker = marker
        self.refused = False
        self._raw: list[str] = []
        self._out: list[str] = []
        self._buf = ""
        self._started = False  # пользователю уже ушёл непробельный текст
        self._strip_ws = False  # после маркера в начале ответа съедаем пробелы и переводы строк

    @property
    def raw(self) -> str:
        """Ответ LLM как есть, с маркером (для трейса Langfuse)."""
        return "".join(self._raw)

    @property
    def text(self) -> str:
        """Уже отданный пользователю текст, без маркера."""
        return "".join(self._out)

    def feed(self, delta: str) -> str:
        self._raw.append(delta)
        self._buf += delta
        out: list[str] = []
        while True:
            i = self._buf.find(self.marker)
            if i < 0:
                break
            before = self._buf[:i]
            if not self._started and not before.strip():
                self.refused = True
                self._strip_ws = True
            else:
                out.append(before)
            self._buf = self._buf[i + len(self.marker) :]
        hold = _held_prefix_len(self._buf, self.marker)
        out.append(self._buf[: len(self._buf) - hold])
        self._buf = self._buf[len(self._buf) - hold :]
        return self._emit("".join(out))

    def flush(self) -> str:
        rest, self._buf = self._buf, ""
        return self._emit(rest)

    def _emit(self, text: str) -> str:
        if self._strip_ws:
            text = text.lstrip()
            if text:
                self._strip_ws = False
        if not self._started:
            if not text.strip():
                # Ведущие пробелы придерживаем: вдруг дальше придёт маркер
                self._buf = text + self._buf
                return ""
            self._started = True
        self._out.append(text)
        return text


def strip_marker(answer: str) -> tuple[str, bool]:
    """Готовый ответ целиком → (текст без маркера, refused)."""
    f = MarkerFilter()
    text = f.feed(answer) + f.flush()
    return text.strip(), f.refused
