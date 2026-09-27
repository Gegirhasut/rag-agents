import asyncio
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

import structlog

from rag_agents.core.config import Settings
from rag_agents.core.db import Database
from rag_agents.core.errors import PermanentError, TransientError
from rag_agents.core.observability import Span, Tracer
from rag_agents.domain.agents import AgentOut
from rag_agents.domain.answers import (
    AnswerUsage,
    Citation,
    DoneEvent,
    ErrorEvent,
    QueryResult,
    RetrievedChunk,
    SourcesEvent,
    StreamEvent,
    TokenEvent,
)
from rag_agents.domain.chats import MessageOut, MessagePair
from rag_agents.domain.enums import MessageRole, MessageStatus
from rag_agents.llm.base import LLMError, LLMProvider, LLMRequest, LLMUsage
from rag_agents.rag.chunking.naive import normalize_for_index
from rag_agents.rag.embeddings.ollama import Embedder
from rag_agents.rag.index.qdrant import QdrantChunkIndex
from rag_agents.rag.prompting.builder import (
    PROMPT_VERSION,
    REFUSAL_TEXT,
    AgentPersona,
    build_citations,
    build_messages,
)
from rag_agents.repositories.agents import AgentRepository
from rag_agents.repositories.chats import ChatRepository
from rag_agents.services.errors import NotFoundError
from rag_agents.services.trace import TraceBus

log = structlog.get_logger()
_TITLE_CHARS = 80


def _ms(since: float) -> int:
    return int((time.monotonic() - since) * 1000)


class QueryService:
    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        index: QdrantChunkIndex,
        llm: LLMProvider,
        trace: TraceBus,
        settings: Settings,
        tracer: Tracer,
    ) -> None:
        self.db = db
        self.tracer = tracer
        self.embedder = embedder
        self.index = index
        self.llm = llm
        self.trace = trace
        self.settings = settings

    async def _agent(self, owner_id: UUID, agent_id: UUID) -> AgentOut:
        async with self.db.session() as s:
            agent = await AgentRepository(s).get(owner_id, agent_id)
        if agent is None:
            raise NotFoundError("agent")
        return agent

    async def history(self, owner_id: UUID, agent_id: UUID) -> list[MessageOut]:
        """Итерация 1: один (последний) чат агента."""
        await self._agent(owner_id, agent_id)
        async with self.db.session() as s:
            repo = ChatRepository(s)
            chat = await repo.latest_chat(agent_id, owner_id)
            return await repo.list_messages(chat.id) if chat else []

    async def ask(
        self, owner_id: UUID, agent_id: UUID, chat_id: UUID | None, question: str
    ) -> MessagePair:
        """Создаёт вопрос и пустой ответ (pending). Генерация стартует при подключении к стриму."""
        agent = await self._agent(owner_id, agent_id)
        await self.trace.emit(
            "query.ask",
            "browser",
            "web",
            f"HTMX: POST вопрос ({len(question)} символов)",
            agent_id=agent_id,
        )
        async with self.db.uow() as uow:
            repo = ChatRepository(uow.session)
            chat = (
                await repo.get_chat(agent_id, owner_id, chat_id)
                if chat_id
                else await repo.latest_chat(agent_id, owner_id)
            )
            if chat_id and chat is None:
                raise NotFoundError("chat")
            if chat is None:
                chat = await repo.create_chat(agent_id, owner_id, question[:_TITLE_CHARS])
            q = await repo.add_message(chat.id, MessageRole.USER, MessageStatus.DONE, question)
            a = await repo.add_message(
                chat.id,
                MessageRole.ASSISTANT,
                MessageStatus.PENDING,
                # В итерации 8 здесь будет вопрос после condense с учётом истории
                standalone_question=question,
                index_id=agent.active_index_id,
            )
            await uow.commit()
        await self.trace.emit(
            "query.saved",
            "web",
            "postgres",
            "INSERT messages: вопрос + пустой ответ (pending); браузер получает HTML с SSE",
            agent_id=agent_id,
        )
        return MessagePair(question=q, answer=a)

    async def stream_answer(
        self, owner_id: UUID, agent_id: UUID, message_id: UUID
    ) -> AsyncIterator[StreamEvent]:
        """Вызывается до начала SSE-ответа: 404 должен уйти обычным HTTP-статусом."""
        agent = await self._agent(owner_id, agent_id)
        async with self.db.uow() as uow:
            repo = ChatRepository(uow.session)
            found = await repo.get_message(agent_id, owner_id, message_id)
            if found is None or found[0].role != MessageRole.ASSISTANT:
                raise NotFoundError("message")
            msg, question = found
            claimed = await repo.claim_for_streaming(message_id)
            await uow.commit()
        if not claimed:
            return self._replay(msg)
        return self._generate(agent, owner_id, msg.chat_id, message_id, question or "")

    async def _replay(self, msg: MessageOut) -> AsyncIterator[StreamEvent]:
        """Повторное подключение EventSource: генерацию заново не запускаем (ARCHITECTURE §6.3)."""
        if msg.status == MessageStatus.DONE:
            citations = [Citation.model_validate(c) for c in msg.citations or []]
            yield DoneEvent(
                result=QueryResult(
                    answer_md=msg.content,
                    refused=bool(msg.refused),
                    citations=citations,
                    usage=AnswerUsage.model_validate(msg.usage) if msg.usage else None,
                )
            )
        else:
            yield ErrorEvent(
                code="not_available",
                message="Ответ уже генерируется или был прерван. Задайте вопрос ещё раз.",
                retryable=True,
            )

    async def feedback(
        self, owner_id: UUID, agent_id: UUID, message_id: UUID, value: Literal[1, -1]
    ) -> MessageOut:
        """👍/👎 под ответом: PG — источник правды, в Langfuse — score на трейсе ответа.

        score_id детерминирован по сообщению: повторный клик перезаписывает оценку, а не дублирует.
        """
        await self._agent(owner_id, agent_id)
        async with self.db.uow() as uow:
            msg = await ChatRepository(uow.session).set_feedback(
                agent_id, owner_id, message_id, value
            )
            if msg is None:
                raise NotFoundError("message")
            await uow.commit()
        if msg.trace_id:
            self.tracer.score(
                trace_id=msg.trace_id,
                name="user_feedback",
                value=1.0 if value > 0 else 0.0,
                score_id=f"feedback-{message_id}",
                data_type="BOOLEAN",
            )
        log.info("query.feedback", message_id=str(message_id), value=value)
        return msg

    async def _generate(
        self, agent: AgentOut, owner_id: UUID, chat_id: UUID, message_id: UUID, question: str
    ) -> AsyncIterator[StreamEvent]:
        t0 = time.monotonic()
        parts: list[str] = []
        citations: list[Citation] = []
        log.info("query.start", agent_id=str(agent.id), message_id=str(message_id))
        root = self._start_trace(agent, owner_id, chat_id, message_id, question)
        generation: Span | None = None
        try:
            await self.trace.emit(
                "query.stream",
                "browser",
                "web",
                "EventSource: SSE-стрим ответа открыт",
                agent_id=agent.id,
            )
            chunks = await self._retrieve(agent, question, root)
            t_retrieval = _ms(t0)

            if not chunks:
                yield DoneEvent(result=await self._refuse(agent, message_id, root, t0))
                return

            citations = build_citations(chunks)
            yield SourcesEvent(citations=citations)

            persona = AgentPersona(agent.name, agent.description, agent.persona_prompt)
            gen = agent.settings.generation
            request = LLMRequest(
                messages=build_messages(persona, question, chunks),
                temperature=gen.temperature,
                max_tokens=gen.max_output_tokens,
            )
            await self._trace_llm_call(agent.id, len(chunks))
            generation = self._start_generation(root, request)
            llm_usage = LLMUsage()
            t_first: int | None = None
            async for chunk in self.llm.stream(request):
                if chunk.delta:
                    if t_first is None:
                        t_first = _ms(t0)
                        # Langfuse считает TTFT от старта generation до completion_start_time
                        generation.update(completion_start_time=datetime.now(UTC))
                        await self._trace_first_token(agent.id, t_first)
                    parts.append(chunk.delta)
                    yield TokenEvent(delta=chunk.delta)
                if chunk.usage:
                    llm_usage = chunk.usage

            answer = "".join(parts).strip()
            usage = AnswerUsage(
                provider=self.llm.name,
                model=self.llm.model,
                reasoning_effort=self.llm.reasoning_effort,
                t_retrieval_ms=t_retrieval,
                t_first_token_ms=t_first,
                t_total_ms=_ms(t0),
                **llm_usage.model_dump(),
            )
            result = QueryResult(
                answer_md=answer,
                refused=answer.startswith(REFUSAL_TEXT[:40]),
                citations=citations,
                usage=usage,
            )
            await self._save(message_id, MessageStatus.DONE, result, root.trace_id)
            self._end_generation(generation, root, result, llm_usage, t_first)
            await self._trace_done(agent.id, usage)
            log.info(
                "query.done",
                message_id=str(message_id),
                ttft_ms=t_first,
                total_ms=usage.t_total_ms,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                reasoning_tokens=usage.reasoning_tokens,
            )
            yield DoneEvent(result=result)
        except (LLMError, TransientError, PermanentError) as e:
            yield await self._fail(e, agent.id, message_id, parts, citations, root, generation)
        except (asyncio.CancelledError, GeneratorExit):
            # Клиент закрыл вкладку: сохраняем то, что успели, и отпускаем отмену дальше
            await asyncio.shield(
                self._save_partial(
                    message_id, MessageStatus.CANCELLED, parts, citations, root.trace_id
                )
            )
            self._end_cancelled(message_id, parts, root, generation)
            raise
        finally:
            # Непредвиденное исключение: закрываем то, что осталось открытым (end идемпотентен)
            if generation is not None:
                generation.end()
            root.end()

    async def _fail(
        self,
        e: LLMError | TransientError | PermanentError,
        agent_id: UUID,
        message_id: UUID,
        parts: list[str],
        citations: list[Citation],
        root: Span,
        generation: Span | None,
    ) -> ErrorEvent:
        log.warning("query.failed", message_id=str(message_id), error=str(e))
        await self.trace.emit(
            "query.failed", "web", "browser", f"Ошибка: {type(e).__name__}", agent_id=agent_id
        )
        await self._save_partial(message_id, MessageStatus.ERROR, parts, citations, root.trace_id)
        status_message = f"{type(e).__name__}: {e}"[:500]
        if generation is not None:
            generation.end(level="ERROR", status_message=status_message)
        root.end(level="ERROR", status_message=status_message)
        return ErrorEvent(
            code="llm_error" if isinstance(e, LLMError) else "retrieval_error",
            message="Не удалось получить ответ. Попробуйте ещё раз.",
            retryable=getattr(e, "retryable", isinstance(e, TransientError)),
        )

    @staticmethod
    def _end_cancelled(
        message_id: UUID, parts: list[str], root: Span, generation: Span | None
    ) -> None:
        if generation is not None:
            generation.end(output="".join(parts), level="WARNING", status_message="cancelled")
        root.end(level="WARNING", status_message="client_cancelled")
        log.info("query.cancelled", message_id=str(message_id))

    def _start_trace(
        self, agent: AgentOut, owner_id: UUID, chat_id: UUID, message_id: UUID, question: str
    ) -> Span:
        """Трейс на вопрос: session = чат, user = владелец, tags = имя агента (§14.2)."""
        return self.tracer.start_trace(
            "query",
            trace_id=self.tracer.trace_id_for(f"query:{message_id}"),
            user_id=str(owner_id),
            session_id=str(chat_id),
            tags=[agent.name],
            input=question,
            metadata={
                "agent_id": str(agent.id),
                "message_id": str(message_id),
                "prompt_version": PROMPT_VERSION,
            },
        )

    def _start_generation(self, root: Span, request: LLMRequest) -> Span:
        return root.child(
            "llm_generate",
            as_type="generation",
            input=[m.model_dump(exclude_none=True) for m in request.messages],
            model=self.llm.model,
            model_parameters={
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "reasoning_effort": self.llm.reasoning_effort,
            },
            metadata={"provider": self.llm.name, "prompt_version": PROMPT_VERSION},
        )

    def _end_generation(
        self,
        generation: Span,
        root: Span,
        result: QueryResult,
        llm_usage: LLMUsage,
        t_first: int | None,
    ) -> None:
        generation.end(
            output=result.answer_md,
            # Ключи совпадают с ценами model definition в Langfuse → он сам считает cost.
            # reasoning уже входит в output_tokens (DeepSeek), поэтому — только в metadata.
            usage_details={
                "input": llm_usage.input_tokens - llm_usage.cached_input_tokens,
                "input_cache_read": llm_usage.cached_input_tokens,
                "output": llm_usage.output_tokens,
            },
            metadata={
                "provider": self.llm.name,
                "prompt_version": PROMPT_VERSION,
                "reasoning_tokens": llm_usage.reasoning_tokens,
                "ttft_ms": t_first,
            },
        )
        root.end(
            output=result.answer_md,
            metadata={"refused": result.refused, "citations": len(result.citations)},
        )

    async def _refuse(
        self, agent: AgentOut, message_id: UUID, root: Span, t0: float
    ) -> QueryResult:
        """Ничего не найдено: отказ без вызова LLM."""
        t_total = _ms(t0)
        usage = AnswerUsage(
            provider="none",
            model="none",
            reasoning_effort=None,
            t_retrieval_ms=t_total,
            t_total_ms=t_total,
        )
        result = QueryResult(answer_md=REFUSAL_TEXT, refused=True, citations=[], usage=usage)
        await self._save(message_id, MessageStatus.DONE, result, root.trace_id)
        root.end(output=REFUSAL_TEXT, metadata={"refused": True, "reason": "no_chunks"})
        await self.trace.emit(
            "query.refused", "web", "browser", "Ничего не найдено → отказ", agent_id=agent.id
        )
        return result

    async def _trace_llm_call(self, agent_id: UUID, n_chunks: int) -> None:
        await self.trace.emit(
            "query.sources",
            "web",
            "browser",
            "SSE event: sources (карточки цитат)",
            agent_id=agent_id,
        )
        await self.trace.emit(
            "query.llm",
            "web",
            "llm",
            f"Промпт {PROMPT_VERSION}: персона + {n_chunks} чанков + вопрос → "
            f"{self.llm.name}/{self.llm.model} (stream=true)",
            agent_id=agent_id,
        )

    async def _trace_first_token(self, agent_id: UUID, t_first: int) -> None:
        await self.trace.emit(
            "query.first_token",
            "llm",
            "web",
            f"Первый токен через {t_first} мс от вопроса",
            agent_id=agent_id,
        )
        await self.trace.emit(
            "query.tokens",
            "web",
            "browser",
            "SSE event: token … token (текст печатается на глазах)",
            agent_id=agent_id,
        )

    async def _trace_done(self, agent_id: UUID, usage: AnswerUsage) -> None:
        await self.trace.emit(
            "query.done",
            "web",
            "postgres",
            f"Ответ сохранён: {usage.input_tokens} вх. / {usage.output_tokens} вых. токенов, "
            f"{usage.t_total_ms} мс",
            agent_id=agent_id,
            usage=usage.model_dump(),
        )
        await self.trace.emit(
            "query.finish",
            "web",
            "browser",
            "SSE event: done (финальный HTML с цитатами)",
            agent_id=agent_id,
        )

    async def _retrieve(self, agent: AgentOut, question: str, root: Span) -> list[RetrievedChunk]:
        """Вопрос → вектор (Ollama) → top-k чанков агента (Qdrant). Каждый шаг — span трейса."""
        emit = self.trace.emit
        collection = await self._collection(agent)
        await emit(
            "query.embed",
            "web",
            "ollama",
            f"POST /api/embed: вопрос → {self.embedder.model}",
            agent_id=agent.id,
        )
        t = time.monotonic()
        with root.child(
            "embed_query",
            as_type="embedding",
            input=question,
            model=self.embedder.model,
        ) as span:
            [vector] = await self.embedder.embed([normalize_for_index(question)])
            span.update(output={"dim": len(vector)})
        await emit(
            "query.embedded",
            "ollama",
            "web",
            f"Вектор вопроса: {len(vector)} измерений ({_ms(t)} мс)",
            agent_id=agent.id,
            vector=[round(v, 4) for v in vector],
        )
        top_k = agent.settings.retrieval.top_k
        await emit(
            "query.search",
            "web",
            "qdrant",
            f"query_points: косинусная близость, top_k={top_k}, filter agent_id",
            agent_id=agent.id,
        )
        t = time.monotonic()
        with root.child(
            "qdrant_search",
            as_type="retriever",
            input={
                "agent_id": str(agent.id),
                "collection": collection,
                "top_k": top_k,
                "mode": "dense",
            },
        ) as span:
            chunks = await self.index.search_dense(collection, agent.id, vector, top_k)
            span.update(
                output=[
                    {
                        "chunk_id": str(c.chunk_id),
                        "document_id": str(c.document_id),
                        "score": round(c.score, 4),
                        "book_title": c.payload.book_title,
                        "chapter": c.payload.chapter_title,
                    }
                    for c in chunks
                ],
                metadata={
                    "agent_id": str(agent.id),
                    "hits": len(chunks),
                    "max_score": round(chunks[0].score, 4) if chunks else None,
                },
            )
        await emit(
            "query.found",
            "qdrant",
            "web",
            f"Найдено чанков: {len(chunks)}"
            + (f", лучший score {chunks[0].score:.3f}" if chunks else "")
            + f" ({_ms(t)} мс)",
            agent_id=agent.id,
            hits=[
                {
                    "id": str(c.chunk_id),
                    "score": round(c.score, 4),
                    "title": c.payload.book_title,
                    "chapter": c.payload.chapter_title,
                }
                for c in chunks
            ],
        )
        return chunks

    async def _collection(self, agent: AgentOut) -> str:
        if agent.active_index_id is None:
            raise PermanentError("no_index", "У агента нет активного индекса")
        async with self.db.session() as s:
            index = await AgentRepository(s).get_index(agent.id, agent.active_index_id)
        if index is None:
            raise PermanentError("no_index", "У агента нет активного индекса")
        return index.collection

    async def _save(
        self, message_id: UUID, status: MessageStatus, result: QueryResult, trace_id: str
    ) -> None:
        async with self.db.uow() as uow:
            await ChatRepository(uow.session).finish_message(
                message_id,
                status=status,
                content=result.answer_md,
                citations=[c.model_dump(mode="json") for c in result.citations],
                refused=result.refused,
                usage=result.usage.model_dump(mode="json") if result.usage else None,
                prompt_version=PROMPT_VERSION,
                trace_id=trace_id,
            )
            await uow.commit()

    async def _save_partial(
        self,
        message_id: UUID,
        status: MessageStatus,
        parts: list[str],
        citations: list[Citation],
        trace_id: str,
    ) -> None:
        async with self.db.uow() as uow:
            await ChatRepository(uow.session).finish_message(
                message_id,
                status=status,
                content="".join(parts),
                citations=[c.model_dump(mode="json") for c in citations] or None,
                prompt_version=PROMPT_VERSION,
                trace_id=trace_id,
            )
            await uow.commit()
