from pathlib import Path

from fastapi.templating import Jinja2Templates

from rag_agents.web.rendering import render_answer

TEMPLATES_DIR = Path(__file__).parent / "templates"

templates = Jinja2Templates(directory=TEMPLATES_DIR)


def _filesize(n: int) -> str:
    size = float(n)
    for unit in ("Б", "КБ", "МБ"):
        if size < 1024:  # noqa: PLR2004
            return f"{size:.0f} {unit}" if unit == "Б" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} ГБ"


templates.env.filters["filesize"] = _filesize
templates.env.globals["render_answer"] = render_answer
