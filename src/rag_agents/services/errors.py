class NotFoundError(Exception):
    """Сущность не найдена или принадлежит другому пользователю (наружу — всегда 404)."""


class ValidationError(Exception):
    """Некорректный ввод пользователя (наружу — 400/422 с текстом).

    message — шаблон с {плейсхолдерами}, params — значения: UI переводит шаблон (core.i18n).
    """

    def __init__(self, message: str, **params: object) -> None:
        self.template = message
        self.params = params
        super().__init__(message.format(**params) if params else message)


class ConflictError(Exception):
    """Операция недопустима в текущем состоянии (наружу — 409)."""
