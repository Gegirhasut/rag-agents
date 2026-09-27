import asyncio

import typer

from rag_agents.container import build_container
from rag_agents.core.config import get_settings
from rag_agents.core.logging import configure_logging

app = typer.Typer(no_args_is_help=True, help="RAG agents admin CLI")


@app.command()
def seed() -> None:
    """Создать seed-пользователя (владельца агентов до появления auth в итерации 2)."""
    settings = get_settings()
    configure_logging(settings.log_level, json=False)

    async def _run() -> None:
        c = build_container(settings)
        try:
            uid = await c.agents.ensure_user(settings.seed_user_email)
            typer.echo(f"seed user: {settings.seed_user_email} ({uid})")
        finally:
            await c.aclose()

    asyncio.run(_run())


@app.command()
def version() -> None:
    typer.echo("rag-agents 0.1.0")


if __name__ == "__main__":
    app()
