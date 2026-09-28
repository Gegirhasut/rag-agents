"""Очистка текста блоков до чанкинга (ARCHITECTURE §7.3, п. 1, 3, 4)."""

import re
import unicodedata

# C0/C1 управляющие (кроме \n и \t), soft hyphen, zero-width, BOM, bidi-метки
_CONTROL = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u00ad\u200b-\u200f\u2028\u2029\u202a-\u202e\u2060\ufeff]"
)
_SPACES = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
_SPACE_AROUND_NL = re.compile(r" *\n *")
# Блок без букв: только пунктуация, цифры, пробелы (номера страниц, «* * *», «———»)
_HAS_LETTER = re.compile(r"[^\W\d_]")
_SCENE_BREAK = re.compile(r"^[*\s·•⁂~—–-]{3,}$")


def clean_text(text: str) -> str:
    """NFC, без управляющих и невидимых символов, пробелы схлопнуты. Кавычки и тире не трогаем."""
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES.sub(" ", text)
    return _SPACE_AROUND_NL.sub("\n", text).strip()


def is_scene_break(text: str) -> bool:
    """«* * *» и подобные разделители сцен: граница блока, но не секции."""
    return bool(_SCENE_BREAK.match(text.strip()))


def is_junk(text: str) -> bool:
    """Блок без единой буквы: номер страницы, разделитель, мусор вёрстки."""
    return not _HAS_LETTER.search(text)
