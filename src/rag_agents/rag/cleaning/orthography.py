"""norm_text для эмбеддинга и BM25 (ARCHITECTURE §7.3, п. 5).

Показ и цитаты идут по исходному тексту: ё и дореформенная орфография там сохраняются.
Для поиска их сводим к современной форме: запрос «мир» должен находить «мiръ».
"""

import re
import unicodedata

_TABLE = str.maketrans(
    {
        "ё": "е",
        "Ё": "Е",
        "ѣ": "е",
        "Ѣ": "Е",
        "і": "и",
        "І": "И",
        "ѳ": "ф",
        "Ѳ": "Ф",
        "ѵ": "и",
        "Ѵ": "И",
    }
)
# Латинская i внутри кириллического слова (частая замена «і» в сканах): «мiръ» → «миръ»
_LATIN_I = re.compile(r"(?<=[А-Яа-яЁё])[iI](?=[А-Яа-яЁё])|(?<=[А-Яа-яЁё])[iI]\b")
# Твёрдый знак на конце слова после согласной: «миръ» → «мир» (но «съел» не трогаем)
_FINAL_HARD_SIGN = re.compile(r"(?<=[бвгджзйклмнпрстфхцчшщБВГДЖЗЙКЛМНПРСТФХЦЧШЩ])[ъЪ]\b")


def norm_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _LATIN_I.sub(lambda m: "И" if m.group() == "I" else "и", text)
    text = text.translate(_TABLE)
    return _FINAL_HARD_SIGN.sub("", text)
