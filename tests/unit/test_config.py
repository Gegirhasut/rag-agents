import pytest

from rag_agents.core.config import Settings


def test_empty_reasoning_effort_means_do_not_send(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_REASONING_EFFORT", "")
    assert Settings(_env_file=None).llm_reasoning_effort is None  # type: ignore[call-arg]


def test_reasoning_effort_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    assert Settings(_env_file=None).llm_reasoning_effort == "high"  # type: ignore[call-arg]


def test_invalid_reasoning_effort_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_REASONING_EFFORT", "extreme")
    with pytest.raises(ValueError, match="llm_reasoning_effort"):
        Settings(_env_file=None)  # type: ignore[call-arg]
