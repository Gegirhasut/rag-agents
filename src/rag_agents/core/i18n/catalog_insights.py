"""RU → EN для UI. Ключ — русская строка как в коде/шаблоне."""

EN: dict[str, str] = {
    # pages/insights*.html
    "Стоимость, скорость и качество ответов. Сводка считается по нашей базе, дерево шагов "
    "каждого запроса — из Langfuse по клику.": (
        "Cost, speed and quality of answers. The summary is computed from our database; "
        "the step tree of each request is loaded from Langfuse on click."
    ),
    "Период": "Period",
    "24 часа": "24 hours",
    "7 дней": "7 days",
    "30 дней": "30 days",
    "Сессия": "Session",
    "сессия": "session",
    "Индексация документа": "Document indexing",
    "Чат": "Chat",
    "{n} трейсов": "{n} traces",
    "Трейс": "Trace",
    "трейс": "trace",
    "Трейс не найден": "Trace not found",
    # fragments/insights_skeleton.html, insights_unavailable.html
    "Загружаю метрики из Langfuse…": "Loading metrics from Langfuse…",
    "Дерево шагов из Langfuse сейчас недоступно.": (
        "The Langfuse step tree is unavailable right now."
    ),
    "Сводка на странице считается по нашей базе и работает; приложение и запись трейсов "
    "не затронуты.": (
        "The summary on this page comes from our database and still works; "
        "the app and trace recording are not affected."
    ),
    "Langfuse не настроен": "Langfuse is not configured",
    "сводка ниже работает, а дерево шагов запросов недоступно.": (
        "the summary below works, but the request step tree is unavailable."
    ),
    "Задайте в": "Set the",
    "переменные": "variables",
    "и": "and",
    "затем перезапустите web и worker": "then restart web and worker",
    "Проверка связи:": "Connection check:",
    "Лимит API Langfuse исчерпан до {until} UTC": (
        "Langfuse API rate limit reached until {until} UTC"
    ),
    # fragments/insights_trace_rows.html
    "Время": "Time",
    "Тип": "Type",
    "Агент": "Agent",
    "Запрос": "Request",
    "Длительность": "Duration",
    "1-й токен": "1st token",
    "Токены": "Tokens",
    "Стоимость": "Cost",
    "Оценка": "Rating",
    "Загружаю трейс…": "Loading trace…",
    "Трейсов пока нет. Задайте агенту вопрос или загрузите документ.": (
        "No traces yet. Ask an agent a question or upload a document."
    ),
    "Ответ не получен (ошибка LLM или поиска)": "No answer received (LLM or search error)",
    # fragments/insights_trace.html
    "Трейс не найден в Langfuse. Возможно, он ещё не дошёл (сеть, фоновый экспорт) или "
    "трейсинг был выключен, когда шёл этот запрос.": (
        "Trace not found in Langfuse. It may not have arrived yet (network, background export), "
        "or tracing was off when this request ran."
    ),
    "Langfuse ещё обрабатывает трейс, обновлю через пару секунд…": (
        "Langfuse is still processing the trace, refreshing in a few seconds…"
    ),
    "ошибка": "error",
    "вся сессия": "whole session",
    "отдельной страницей": "open as page",
    "ожидание первого токена": "waiting for first token",
    "модель": "model",
    "первый токен": "first token",
    "токены": "tokens",
    "источник": "source",
    "деталей нет": "no details",
    "Вопрос": "Question",
    "Ответ ({n} симв.)": "Answer ({n} chars)",
    "Промпт LLM: {n} сообщения, {chars} символов": "LLM prompt: {n} messages, {chars} characters",
    # fragments/insights_overview.html
    "за {period}": "last {period}",
    "обновлено": "updated",
    "раз в минуту · источник — наша БД": "every minute · source: our DB",
    "Вопросы": "Questions",
    "{n} вызовов LLM": "{n} LLM calls",
    "{cost} за ответ": "{cost} per answer",
    "Ответ целиком": "Full answer",
    "Time to first token: от запроса к LLM до первого токена ответа": (
        "Time to first token: from the LLM request to the first answer token"
    ),
    "Первый токен": "First token",
    "Оценки": "Ratings",
    "доля 👍": "share of 👍",
    "Индексации · ошибки": "Indexing · errors",
    "Вопросы и стоимость": "Questions and cost",
    "по часам": "hourly",
    "по дням": "daily",
    "ваше время": "your time zone",
    "Вопросы и стоимость по времени": "Questions and cost over time",
    "За период вопросов не было.": "No questions in this period.",
    "Где тратится время": "Where the time goes",
    "p50, риска — p95": "p50, tick = p95",
    "Ответ на вопрос": "Answering a question",
    "{n} шт.": "{n} items",
    "llm: до первого токена": "llm: until first token",
    "llm: генерация ответа": "llm: answer generation",
    "Нет данных.": "No data.",
    "Агенты": "Agents",
    "Ответ p50 / p95": "Answer p50 / p95",
    "удалён": "deleted",
    "нет оценок": "no ratings",
    "За период нет вопросов.": "No questions in this period.",
    "Последние запросы": "Recent requests",
    "клик — дерево шагов, промпт и найденные чанки": (
        "click for the step tree, prompt and retrieved chunks"
    ),
    "Модели LLM": "LLM models",
    "Модель": "Model",
    "Вызовы": "Calls",
    "tokens|Вход": "Input",
    "Выход": "Output",
    "Стоимость — по": "Cost is based on",
    "с учётом тарифов peak/off-peak DeepSeek; та же сумма уходит в Langfuse.": (
        "including DeepSeek peak/off-peak rates; the same amount is sent to Langfuse."
    ),
}

# Строки для static/*.js (window.t)
JS_EN: dict[str, str] = {
    "Вопросы (вызовы LLM)": "Questions (LLM calls)",
    "Стоимость, $": "Cost, $",
    "{n} вопр.": "{n} questions",
}
