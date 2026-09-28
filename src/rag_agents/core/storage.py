import contextlib
import hashlib
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import anyio


class FileTooLargeError(Exception):
    pass


@dataclass(frozen=True)
class StoredFile:
    key: str
    size: int
    sha256: bytes


class LocalFileStorage:
    """Оригиналы загрузок на volume. Имя файла пользователя в путях не участвует."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, key: str) -> Path:
        return self.root / key

    async def save_stream(
        self,
        agent_id: UUID,
        document_id: UUID,
        ext: str,
        chunks: AsyncIterator[bytes],
        max_bytes: int,
    ) -> StoredFile:
        key = f"{agent_id}/{document_id}/original.{ext}"
        target = anyio.Path(self.path(key))
        await target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        try:
            async with await anyio.open_file(target, "wb") as f:
                async for chunk in chunks:
                    size += len(chunk)
                    if size > max_bytes:
                        raise FileTooLargeError(f"file exceeds {max_bytes} bytes")  # noqa: TRY301  частичный файл удаляет except ниже
                    digest.update(chunk)
                    await f.write(chunk)
        except BaseException:
            await target.unlink(missing_ok=True)
            raise
        return StoredFile(key=key, size=size, sha256=digest.digest())

    async def delete(self, key: str) -> None:
        await anyio.Path(self.path(key)).unlink(missing_ok=True)

    async def delete_agent_dir(self, agent_id: UUID) -> None:
        """Все файлы агента: <root>/<agent_id>/… (очистка удалённого агента)."""
        target = self.path(str(agent_id))
        await anyio.to_thread.run_sync(lambda: shutil.rmtree(target, ignore_errors=True))

    async def delete_document_dir(self, key: str) -> None:
        """Файл документа и его пустой каталог <agent>/<document>/."""
        await self.delete(key)
        with contextlib.suppress(OSError):
            await anyio.Path(self.path(key)).parent.rmdir()
