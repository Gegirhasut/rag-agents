from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Self

import structlog
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

log = structlog.get_logger()

AfterCommit = Callable[[], Awaitable[None] | None]


def make_engine(url: str, pool_size: int = 5) -> AsyncEngine:
    return create_async_engine(url, pool_size=pool_size, max_overflow=5, pool_pre_ping=True)


class UnitOfWork:
    """Транзакция сервиса. Коммитит сервис; колбэки on_commit выполняются после COMMIT.

    Так публикация в очередь никогда не опережает запись в БД (ARCHITECTURE §4).
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = session_factory
        self._after_commit: list[AfterCommit] = []
        self.session: AsyncSession

    async def __aenter__(self) -> Self:
        self.session = self._factory()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            await self.session.rollback()
        await self.session.close()

    def on_commit(self, fn: AfterCommit) -> None:
        self._after_commit.append(fn)

    async def commit(self) -> None:
        await self.session.commit()
        callbacks, self._after_commit = self._after_commit, []
        for fn in callbacks:
            result = fn()
            if result is not None:
                await result


class Database:
    def __init__(self, url: str, pool_size: int = 5) -> None:
        self.engine = make_engine(url, pool_size)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    def uow(self) -> UnitOfWork:
        return UnitOfWork(self.session_factory)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as s:
            yield s

    async def dispose(self) -> None:
        await self.engine.dispose()
