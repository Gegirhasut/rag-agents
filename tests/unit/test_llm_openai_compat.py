import json

import httpx
import pytest

from rag_agents.llm.base import LLMError, LLMMessage, LLMRequest
from rag_agents.llm.openai_compat import OpenAICompatProvider

REQ = LLMRequest(messages=[LLMMessage(role="user", content="Сколько будет 17*23?")])


def sse(*events: dict[str, object]) -> bytes:
    lines = [f"data: {json.dumps(e, ensure_ascii=False)}\n\n" for e in events]
    return ("".join(lines) + ": keep-alive\n\ndata: [DONE]\n\n").encode()


STREAM = sse(
    {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
    {"choices": [{"delta": {"reasoning_content": "17*23 = 391"}}]},
    {"choices": [{"delta": {"content": "39"}}]},
    {"choices": [{"delta": {"content": "1"}, "finish_reason": "stop"}]},
    {
        "choices": [],
        "usage": {
            "prompt_tokens": 45,
            "completion_tokens": 19,
            "prompt_tokens_details": {"cached_tokens": 32},
            "completion_tokens_details": {"reasoning_tokens": 17},
        },
    },
)


def provider(handler: httpx.MockTransport, effort: str | None) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        name="deepseek",
        base_url="https://api.test",
        api_key="k",
        model="deepseek-flash",
        reasoning_effort=effort,
        transport=handler,
    )


@pytest.mark.parametrize("effort", ["low", "high", None])
async def test_streams_content_hides_reasoning_and_reports_usage(effort: str | None) -> None:
    sent: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer k"
        return httpx.Response(200, content=STREAM, headers={"content-type": "text/event-stream"})

    chunks = [c async for c in provider(httpx.MockTransport(handle), effort).stream(REQ)]

    assert "".join(c.delta for c in chunks) == "391"
    usage = next(c.usage for c in chunks if c.usage)
    assert (usage.input_tokens, usage.output_tokens) == (45, 19)
    assert (usage.reasoning_tokens, usage.cached_input_tokens) == (17, 32)
    body = sent[0]
    assert body["model"] == "deepseek-flash"
    assert body["stream"] is True
    if effort is None:
        assert "reasoning_effort" not in body
    else:
        assert body["reasoning_effort"] == effort


@pytest.mark.parametrize(("status", "retryable"), [(429, True), (503, True), (401, False)])
async def test_http_errors_are_classified(status: int, retryable: bool) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(status, text="nope"))
    with pytest.raises(LLMError) as e:
        async for _ in provider(transport, "low").stream(REQ):
            pass
    assert e.value.retryable is retryable
    assert e.value.status == status
