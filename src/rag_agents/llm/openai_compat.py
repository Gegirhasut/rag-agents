import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from rag_agents.llm.base import LLMChunk, LLMError, LLMRequest, LLMUsage

log = structlog.get_logger()

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def parse_usage(raw: dict[str, Any]) -> LLMUsage:
    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    cached = prompt_details.get("cached_tokens") or raw.get("prompt_cache_hit_tokens") or 0
    return LLMUsage(
        input_tokens=raw.get("prompt_tokens", 0),
        output_tokens=raw.get("completion_tokens", 0),
        reasoning_tokens=completion_details.get("reasoning_tokens", 0),
        cached_input_tokens=cached,
    )


class OpenAICompatProvider:
    """Chat Completions со стримингом поверх httpx: DeepSeek, OpenAI, Ollama (/v1).

    SDK openai не используем: нужен полный контроль над таймаутами, reasoning-полями
    и разбором SSE, а зависимость ради одного эндпоинта избыточна.
    """

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str | None,
        model: str,
        reasoning_effort: str | None,
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.reasoning_effort = reasoning_effort
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(read_timeout_s, connect=connect_timeout_s),
            transport=transport,
        )

    def build_payload(self, req: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump(exclude_none=True) for m in req.messages],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    async def stream(self, req: LLMRequest) -> AsyncIterator[LLMChunk]:
        payload = self.build_payload(req)
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.is_error:
                    body = (await resp.aread()).decode("utf-8", "replace")[:300]
                    raise LLMError(
                        f"{self.name} HTTP {resp.status_code}: {body}",
                        retryable=resp.status_code in _RETRYABLE_STATUS,
                        status=resp.status_code,
                    )
                async for line in resp.aiter_lines():
                    chunk = self._parse_line(line)
                    if chunk is not None:
                        yield chunk
        except httpx.TimeoutException as e:
            raise LLMError(f"{self.name} timeout: {e!r}", retryable=True) from e
        except httpx.TransportError as e:
            raise LLMError(f"{self.name} transport error: {e!r}", retryable=True) from e

    @staticmethod
    def _parse_line(line: str) -> LLMChunk | None:
        if not line.startswith("data:"):
            return None  # пустые строки и SSE-комментарии (keep-alive)
        data = line[5:].strip()
        if not data or data == "[DONE]":
            return None
        event = json.loads(data)
        usage = parse_usage(event["usage"]) if event.get("usage") else None
        choices = event.get("choices") or []
        if not choices:
            return LLMChunk(usage=usage) if usage else None
        choice = choices[0]
        delta = (choice.get("delta") or {}).get("content") or ""
        finish = choice.get("finish_reason")
        if not delta and not finish and usage is None:
            return None  # чистый reasoning_content или роль — наружу не отдаём
        return LLMChunk(delta=delta, finish_reason=finish, usage=usage)

    async def aclose(self) -> None:
        await self._client.aclose()
