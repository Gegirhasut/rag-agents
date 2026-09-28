from typing import Protocol

from rag_agents.domain.tasks import DeleteDocumentTask, EmbedBatchTask, ParseTask, PurgeAgentTask


class TaskPublisher(Protocol):
    """Публикация задач в очереди. Вызывается только после коммита (uow.on_commit)."""

    def publish_parse(self, task: ParseTask) -> None: ...

    def publish_embed(self, tasks: list[EmbedBatchTask]) -> None: ...

    def publish_delete_document(self, task: DeleteDocumentTask) -> None: ...

    def publish_purge_agent(self, task: PurgeAgentTask) -> None: ...
