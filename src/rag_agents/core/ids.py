from datetime import UTC, datetime
from uuid import UUID

import uuid_utils


def uuid7() -> UUID:
    """UUID v7: упорядочен по времени — дружелюбнее к B-tree индексам, чем v4."""
    return UUID(str(uuid_utils.uuid7()))


def utcnow() -> datetime:
    return datetime.now(UTC)
