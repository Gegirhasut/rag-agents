"""Replay DLQ (ARCHITECTURE §10.3): сообщения из <queue>.dlq обратно в рабочую очередь.

Задачи идемпотентны, поэтому повтор безопасен: уже сделанная работа станет no-op.
"""

from dataclasses import dataclass
from typing import Any

from kombu import Connection, Producer

from rag_agents.workers.celery_app import QUEUE_NAMES, TASKS_EXCHANGE, celery_app


@dataclass(frozen=True)
class ReplayResult:
    queue: str
    replayed: int


def replay(queue: str, limit: int | None = None) -> ReplayResult:
    if queue not in QUEUE_NAMES:
        raise ValueError(f"неизвестная очередь {queue}; есть: {', '.join(QUEUE_NAMES)}")
    replayed = 0
    with Connection(celery_app.conf.broker_url) as conn:
        # Any: у типов kombu StdChannel нет basic_get/basic_ack, хотя у AMQP-канала они есть
        channel: Any = conn.channel()
        producer = Producer(channel, exchange=TASKS_EXCHANGE)
        while limit is None or replayed < limit:
            msg = channel.basic_get(f"{queue}.dlq", no_ack=False)
            if msg is None:
                break
            headers = dict(msg.headers or {})
            # Счётчик ретраев Celery обнуляем: иначе исчерпанная задача получит одну попытку
            headers["retries"] = 0
            headers.pop("x-death", None)
            producer.publish(
                msg.body,
                routing_key=queue,
                headers=headers,
                content_type=msg.content_type,
                content_encoding=msg.content_encoding,
                correlation_id=msg.properties.get("correlation_id"),
                delivery_mode=2,
            )
            channel.basic_ack(msg.delivery_tag)
            replayed += 1
    return ReplayResult(queue=queue, replayed=replayed)
