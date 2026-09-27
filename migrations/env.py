import asyncio

from alembic import context
from sqlalchemy.engine import Connection

from rag_agents.core.config import get_settings
from rag_agents.core.db import make_engine
from rag_agents.models import entities  # noqa: F401  регистрирует таблицы в metadata
from rag_agents.models.base import Base

target_metadata = Base.metadata


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    url = context.config.get_main_option("sqlalchemy.url") or get_settings().database_url
    engine = make_engine(url, pool_size=1)
    async with engine.connect() as conn:
        await conn.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    raise SystemExit("offline mode is not supported")
asyncio.run(run_async_migrations())
