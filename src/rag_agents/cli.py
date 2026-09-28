import asyncio
from collections.abc import Awaitable, Callable
from typing import Annotated

import typer

from rag_agents.container import Container, build_container
from rag_agents.core.config import get_settings
from rag_agents.core.logging import configure_logging
from rag_agents.services.errors import NotFoundError, ValidationError

app = typer.Typer(no_args_is_help=True, help="RAG agents admin CLI")
user_app = typer.Typer(no_args_is_help=True, help="Пользователи: создание и смена пароля")
app.add_typer(user_app, name="user")


def _run_with_container(fn: Callable[[Container], Awaitable[None]]) -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json=False)

    async def _run() -> None:
        c = build_container(settings)
        try:
            await fn(c)
        finally:
            await c.aclose()

    asyncio.run(_run())


PasswordOpt = Annotated[
    str,
    typer.Option(
        prompt=True,
        confirmation_prompt=True,
        hide_input=True,
        envvar="RAG_USER_PASSWORD",
        help="Пароль. Без опции спрашивается интерактивно.",
    ),
]


@user_app.command("create")
def user_create(
    email: str,
    password: PasswordOpt,
    admin: Annotated[bool, typer.Option("--admin", help="Доступ к /system")] = False,
) -> None:
    """Создать пользователя: rag-agents user create me@example.com --admin"""

    async def _create(c: Container) -> None:
        try:
            user = await c.auth.create_user(email, password, is_admin=admin)
        except ValidationError as e:
            raise typer.BadParameter(str(e)) from e
        typer.echo(f"создан: {user.email} ({user.id}){' admin' if user.is_admin else ''}")

    _run_with_container(_create)


@user_app.command("set-password")
def user_set_password(
    email: str,
    password: PasswordOpt,
    admin: Annotated[
        bool | None, typer.Option("--admin/--no-admin", help="Заодно поменять роль")
    ] = None,
) -> None:
    """Сменить пароль (все сессии пользователя завершаются)."""

    async def _set(c: Container) -> None:
        try:
            user = await c.auth.set_password(email, password, is_admin=admin)
        except ValidationError as e:
            raise typer.BadParameter(str(e)) from e
        except NotFoundError as e:
            raise typer.BadParameter(f"нет пользователя {email}") from e
        typer.echo(f"пароль обновлён: {user.email}{' admin' if user.is_admin else ''}")

    _run_with_container(_set)


@app.command()
def seed(
    password: Annotated[
        str | None,
        typer.Option(envvar="SEED_USER_PASSWORD", help="Пароль seed-администратора"),
    ] = None,
) -> None:
    """Seed-администратор SEED_USER_EMAIL (владелец агентов итерации 1). С паролем — можно войти."""

    async def _seed(c: Container) -> None:
        email = c.settings.seed_user_email
        uid = await c.agents.ensure_user(email)
        if password:
            await c.auth.set_password(email, password, is_admin=True)
            typer.echo(f"seed admin: {email} ({uid}), пароль задан")
        else:
            typer.echo(
                f"seed user: {email} ({uid}); пароля нет — задайте: "
                f"rag-agents user set-password {email} --admin"
            )

    _run_with_container(_seed)


dlq_app = typer.Typer(no_args_is_help=True, help="Dead letter queues")
app.add_typer(dlq_app, name="dlq")


@dlq_app.command("replay")
def dlq_replay(
    queue: str,
    limit: Annotated[int | None, typer.Option(help="Сколько сообщений (по умолчанию все)")] = None,
) -> None:
    """Переложить сообщения из <queue>.dlq обратно в <queue>: rag-agents dlq replay ingest.embed"""
    from rag_agents.workers.dlq import replay  # noqa: PLC0415  kombu нужен только этой команде

    try:
        result = replay(queue, limit)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    typer.echo(f"{result.queue}.dlq → {result.queue}: {result.replayed} сообщений")


@app.command()
def chaos_embed() -> None:
    """Следующий батч эмбеддинга упадёт с внутренней ошибкой → ingest.embed.dlq (только dev)."""
    from rag_agents.services.ingest import CHAOS_EMBED_KEY  # noqa: PLC0415

    async def _set(c: Container) -> None:
        if c.settings.app_env == "prod":
            raise typer.BadParameter("в prod хаос-переключатель выключен")
        await c.redis.set(CHAOS_EMBED_KEY, "1", ex=3600)
        typer.echo("следующий батч эмбеддинга упадёт (ключ действует 1 час)")

    _run_with_container(_set)


@app.command()
def version() -> None:
    typer.echo("rag-agents 0.1.0")


if __name__ == "__main__":
    app()
