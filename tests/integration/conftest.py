import os
import uuid
from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient

from rag_agents.core.db import Database

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+asyncpg://rag_test:rag_test@127.0.0.1:15432/rag_test"
)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:16333")


@pytest.fixture(scope="session")
async def db() -> AsyncIterator[Database]:
    database = Database(DATABASE_URL, pool_size=2)
    yield database
    await database.dispose()


@pytest.fixture(scope="session")
async def qdrant() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(url=QDRANT_URL)
    yield client
    await client.close()


@pytest.fixture
def collection_prefix() -> str:
    return f"t{uuid.uuid4().hex[:8]}_"
