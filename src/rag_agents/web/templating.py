from collections.abc import Iterable
from pathlib import Path

from fastapi.templating import Jinja2Templates
from jinja2.utils import htmlsafe_json_dumps
from markupsafe import Markup
from pydantic import BaseModel

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


def _ms(v: float | None) -> str:
    if v is None:
        return "—"
    if v < 1000:  # noqa: PLR2004  порог перехода мс → с
        return f"{v:.0f} мс"
    return f"{v / 1000:.1f} с"


def _usd(v: float | None) -> str:
    if v is None:
        return "—"
    if v == 0:
        return "$0"
    return f"${v:.4f}" if v < 0.01 else f"${v:.2f}"  # noqa: PLR2004  центы и доли цента


def _num(v: int | float | None) -> str:
    return "—" if v is None else f"{v:,.0f}".replace(",", "\u202f")


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.0f} %"


def _models_json(items: Iterable[BaseModel]) -> Markup:
    """Список Pydantic-моделей → JSON, безопасный внутри <script> (данные для графиков)."""
    return htmlsafe_json_dumps([m.model_dump(mode="json") for m in items])


templates.env.filters["filesize"] = _filesize
templates.env.filters["models_json"] = _models_json
templates.env.filters["ms"] = _ms
templates.env.filters["usd"] = _usd
templates.env.filters["num"] = _num
templates.env.filters["pct"] = _pct
templates.env.globals["render_answer"] = render_answer
