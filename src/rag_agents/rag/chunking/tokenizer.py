from functools import cache
from pathlib import Path
from typing import Protocol

from tokenizers import Tokenizer


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def count_batch(self, texts: list[str]) -> list[int]: ...


class HFTokenCounter:
    """Токенизатор bge-m3 (XLM-R SentencePiece): лимиты чанка считаются в токенах модели."""

    def __init__(self, tokenizer_json: Path) -> None:
        self._tok = Tokenizer.from_file(str(tokenizer_json))
        self._tok.no_truncation()
        self._tok.no_padding()

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def count_batch(self, texts: list[str]) -> list[int]:
        return [len(e.ids) for e in self._tok.encode_batch(texts, add_special_tokens=False)]


@cache
def load_token_counter(tokenizer_json: Path) -> HFTokenCounter:
    """Один экземпляр на процесс: словарь XLM-R в памяти занимает ~240 МБ.

    Воркер вызывает это в родителе Celery до fork, и дочерние процессы делят страницы
    токенизатора через copy-on-write вместо загрузки своей копии.
    """
    return HFTokenCounter(tokenizer_json)
