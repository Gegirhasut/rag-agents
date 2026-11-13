# ruff: noqa: E501  ключи каталога — исходные строки целиком
"""RU → EN для UI. Ключ — русская строка как в коде/шаблоне."""

EN: dict[str, str] = {
    # base.html
    "RAG-агенты": "RAG Agents",
    "Аналитика": "Insights",
    "Качество": "Quality",
    "Под капотом": "Under the hood",
    "API-ключи": "API keys",
    "Выйти": "Log out",
    # login
    "Вход": "Sign in",
    "Пароль": "Password",
    "Войти": "Sign in",
    "Пользователя создаёт администратор:": "Users are created by an administrator:",
    "Неверный email или пароль": "Invalid email or password",
    # agents list / new agent
    "Агенты": "Agents",
    "Создать агента": "Create agent",
    "документов:": "documents:",
    "Агентов пока нет. Создайте первого — например, «Внутренний мир Льва Толстого».": (
        "No agents yet. Create your first one, e.g. “The inner world of Leo Tolstoy”."
    ),
    "Новый агент": "New agent",
    "Название": "Name",
    "Внутренний мир Льва Толстого": "The inner world of Leo Tolstoy",
    "Описание": "Description",
    "Персона (необязательно)": "Persona (optional)",
    "Отвечай как исследователь творчества Толстого…": "Answer as a Tolstoy scholar…",
    "Задаёт тон ответов. Правила «только по источникам» персона не отменяет.": (
        "Sets the tone of answers. The persona does not override the “sources only” rules."
    ),
    "Создать": "Create",
    "Отмена": "Cancel",
    # agent detail
    "агенты": "agents",
    "смотреть под капотом (откроется рядом)": "look under the hood (opens in a new tab)",
    "Документы": "Documents",
    "Загрузить": "Upload",
    ".txt (UTF-8, cp1251, koi8-r), .fb2, .fb2.zip, .epub, .pdf (с текстовым слоем), .docx — до 100 МБ.": (
        ".txt (UTF-8, cp1251, koi8-r), .fb2, .fb2.zip, .epub, .pdf (with a text layer), "
        ".docx — up to 100 MB."
    ),
    "Вопрос агенту": "Ask the agent",
    "чат в аналитике": "chat in Insights",
    "Ответ прерван ({status})": "Answer interrupted ({status})",
    "В чём смысл жизни?": "What is the meaning of life?",
    "Спросить": "Ask",
    # answers, sources, usage (macros + fragments)
    "Источник": "Source",
    "Источники": "Sources",
    "гл.": "ch.",
    "Ответ оборван: модель упёрлась в лимит токенов (reasoning {n}). Увеличьте LLM_REASONING_BUDGET_TOKENS или max_output_tokens агента.": (
        "Answer truncated: the model hit the token limit (reasoning {n}). "
        "Increase LLM_REASONING_BUDGET_TOKENS or the agent's max_output_tokens."
    ),
    "поиск": "retrieval",
    "мс": "ms",
    "с": "s",
    "первый токен": "first token",
    "всего": "total",
    "токены": "tokens",
    "ответ без вызова LLM": "answered without an LLM call",
    "Шаги, тайминги, промпт и стоимость этого ответа": (
        "Steps, timings, prompt and cost of this answer"
    ),
    "трейс": "trace",
    "вопрос": "question",
    "индексация": "indexing",
    "полезный ответ": "helpful answer",
    "плохой ответ": "bad answer",
    "Полезный ответ": "Helpful answer",
    "Плохой ответ": "Bad answer",
    "Ответ не найден в материалах агента.": "No answer found in the agent's materials.",
    "ищу в источниках…": "searching the sources…",
    "Ответ не получен": "No answer received",
    "Ответ уже генерируется или был прерван. Задайте вопрос ещё раз.": (
        "The answer is already being generated or was interrupted. Please ask again."
    ),
    "Не удалось получить ответ. Попробуйте ещё раз.": "Could not get an answer. Please try again.",
    # documents table / document page
    "Файл": "File",
    "Размер": "Size",
    "Статус": "Status",
    "статус": "status",
    "чанков": "chunks",
    "батчей": "batches",
    "Обработать заново": "Process again",
    "Удалить документ": "Delete document",
    "Удалить «{name}» и его фрагменты из индекса?": (
        "Delete “{name}” and its chunks from the index?"
    ),
    "Документов пока нет.": "No documents yet.",
    "Оглавление": "Table of contents",
    "из файла (TOC, стили, разметка)": "from the file (TOC, styles, markup)",
    "эвристика заголовков": "heading heuristics",
    "не найдено — одна секция": "not found — single section",
    "Кодировка": "Encoding",
    "Страниц": "Pages",
    "Язык": "Language",
    "Фрагментов": "Chunks",
    "батчей эмбеддинга": "embedding batches",
    "timings|Время": "Timings",
    "(без заголовка)": "(untitled)",
    "фр.": "chunks",
    "ток.": "tok.",
    "стр.": "p.",
    "Фрагменты {a}–{b} из {total}": "Chunks {a}–{b} of {total}",
    "Фрагментов нет": "No chunks",
    "документ ещё обрабатывается": "the document is still being processed",
    # upload / document errors
    "«{name}» уже загружен в этого агента — пропущен": (
        "“{name}” is already uploaded to this agent — skipped"
    ),
    "«{name}»: лимит загрузок, повторите через {s} с": (
        "“{name}”: upload limit reached, retry in {s} s"
    ),
    "Неподдерживаемый формат: {suffix}. Можно: .txt, .fb2, .fb2.zip, .epub, .pdf, .docx": (
        "Unsupported format: {suffix}. Allowed: .txt, .fb2, .fb2.zip, .epub, .pdf, .docx"
    ),
    "без расширения": "no extension",
    "Файл больше {mb} МБ": "File is larger than {mb} MB",
    "Файл пустой": "File is empty",
    "Содержимое файла не похоже на {fmt}": "File content does not look like {fmt}",
    "Пользователь {email} уже существует": "User {email} already exists",
    "Пароль не может быть пустым": "Password cannot be empty",
    "Повторить можно только документ в статусе failed": "Only a failed document can be retried",
    "Сервис индексации недоступен": "Indexing service is unavailable",
    "Внутренняя ошибка обработки": "Internal processing error",
    "Документ или агент удалены": "The document or agent has been deleted",
    "Файл загрузки не найден — загрузите заново": "Uploaded file not found — please upload again",
    "В документе не найдено текста": "No text found in the document",
    "Документ больше 20000 фрагментов — разбейте файл на части": (
        "Document exceeds 20000 chunks — split the file into parts"
    ),
    "Фрагменты документа не найдены": "Document chunks not found",
    "У агента нет активного индекса": "The agent has no active index",
    "Ollama вернула векторы неожиданной формы": "Ollama returned vectors of an unexpected shape",
    "Файл повреждён: DOCX не открывается": "Corrupted file: the DOCX cannot be opened",
    "Файл повреждён: не найден OPF": "Corrupted file: OPF not found",
    "Файл повреждён: нет rootfile в container.xml": "Corrupted file: no rootfile in container.xml",
    "В архиве нет файла .fb2": "The archive contains no .fb2 file",
    "Файл повреждён: некорректный XML": "Corrupted file: invalid XML",
    "Файл повреждён: PDF не открывается": "Corrupted file: the PDF cannot be opened",
    "PDF защищён паролем": "The PDF is password-protected",
    "В PDF нет страниц": "The PDF has no pages",
    "Файл повреждён: не удалось открыть архив": "Corrupted file: the archive cannot be opened",
    "Архив распаковывается в слишком большой объём": "The archive unpacks to too large a size",
    "Файл повреждён: пустой XML": "Corrupted file: empty XML",
    "Не удалось определить кодировку текста": "Could not detect the text encoding",
    # API keys
    "Ключ даёт доступ к JSON API от вашего имени:": "A key grants access to the JSON API on your behalf:",
    "Документация —": "Docs:",
    "Название, например «ноутбук»": "Name, e.g. “laptop”",
    "Выпустить ключ": "Issue key",
    "Ключ «{name}» выпущен. Скопируйте его сейчас — больше он показан не будет:": (
        "Key “{name}” issued. Copy it now — it will not be shown again:"
    ),
    "Префикс": "Prefix",
    "Создан": "Created",
    "Последнее использование": "Last used",
    "Отозвать ключ «{name}»? Скрипты с ним перестанут работать.": (
        "Revoke key “{name}”? Scripts using it will stop working."
    ),
    "Отозвать": "Revoke",
    "отозван": "revoked",
    "Ключей пока нет.": "No keys yet.",
    "Название ключа — от 1 до 80 символов": "Key name must be 1 to 80 characters",
    # error pages / HTTP errors
    "к списку агентов": "back to agents",
    "Некорректные данные формы": "Invalid form data",
    "Не найдено": "Not found",
    "Страница не найдена": "Page not found",
    "Агент не найден": "Agent not found",
    "Агент или чат не найден": "Agent or chat not found",
    "Сообщение не найдено": "Message not found",
    "Вопрос должен быть от 2 до 2000 символов": "The question must be 2 to 2000 characters",
    "CSRF-токен неверный или отсутствует. Обновите страницу.": (
        "CSRF token is invalid or missing. Please reload the page."
    ),
    "Слишком много запросов. Повторите через {s} с.": "Too many requests. Retry in {s} s.",
    # templating filters (units)
    "Б": "B",
    "КБ": "KB",
    "МБ": "MB",
    "ГБ": "GB",
}

# Строки для static/*.js (window.t)
JS_EN: dict[str, str] = {}
