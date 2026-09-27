from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agents.domain.system import TableStats


class PgStatsRepository:
    """Служебная статистика PostgreSQL (каталоги pg_*), без данных пользователей."""

    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def version(self) -> str:
        v = await self.s.scalar(text("SHOW server_version"))
        return str(v)

    async def db_size(self) -> str:
        v = await self.s.scalar(text("SELECT pg_size_pretty(pg_database_size(current_database()))"))
        return str(v)

    async def tables(self) -> list[TableStats]:
        # n_live_tup — оценка из статистики автовакуума: дёшево и без seq scan больших таблиц
        rows = await self.s.execute(
            text(
                "SELECT relname, n_live_tup, pg_size_pretty(pg_total_relation_size(relid)) "
                "FROM pg_stat_user_tables ORDER BY pg_total_relation_size(relid) DESC"
            )
        )
        return [TableStats(name=r[0], rows=int(r[1]), size=str(r[2])) for r in rows]
