class TransientError(Exception):
    """Временный сбой инфраструктуры (Ollama, Qdrant, сеть) — задачу можно ретраить."""


class PermanentError(Exception):
    """Ошибка входных данных — ретрай не поможет, документ сразу получает failed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
