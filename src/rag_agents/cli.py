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
        help="Пароль (≥ 8 символов). Без опции спрашивается интерактивно.",
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


@app.command()
def version() -> None:
    typer.echo("rag-agents 0.1.0")


if __name__ == "__main__":
    app()
