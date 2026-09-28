from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from rag_agents.llm.base import LLMUsage
from rag_agents.llm.prices import PriceTable

PRICES = PriceTable.load(Path("configs/llm_prices.yaml"))
USAGE = LLMUsage(
    input_tokens=2443, cached_input_tokens=128, output_tokens=744, reasoning_tokens=100
)
MONDAY_PEAK = datetime(2026, 9, 28, 7, 30, tzinfo=UTC)  # пн, 06:00–10:00 UTC


def test_peak_cost_matches_langfuse_model_definition() -> None:
    """Сверено с Langfuse на smoke-трейсе 2026-09-27: $0.001588068 по peak-ценам."""
    c = PRICES.cost("deepseek", "deepseek-flash", USAGE, MONDAY_PEAK)
    assert c is not None
    assert c.peak
    assert c.total == pytest.approx(0.001588068)
    assert c.as_langfuse()["input"] == pytest.approx(2315 * 0.30 / 1e6)


@pytest.mark.parametrize(
    ("at", "peak"),
    [
        (datetime(2026, 9, 28, 1, 0, tzinfo=UTC), True),  # начало окна включительно
        (datetime(2026, 9, 28, 4, 0, tzinfo=UTC), False),  # конец окна исключительно
        (datetime(2026, 9, 28, 5, 0, tzinfo=UTC), False),  # между окнами
        (datetime(2026, 9, 28, 20, 0, tzinfo=UTC), False),  # вечер
        (datetime(2026, 9, 27, 7, 30, tzinfo=UTC), False),  # воскресенье
        # 10:30 по Москве = 07:30 UTC понедельника → peak
        (datetime(2026, 9, 28, 10, 30, tzinfo=timezone(timedelta(hours=3))), True),
    ],
)
def test_off_peak_is_half_price(at: datetime, peak: bool) -> None:
    c = PRICES.cost("deepseek", "deepseek-flash", USAGE, at)
    assert c is not None
    assert c.peak is peak
    assert c.total == pytest.approx(0.001588068 * (1 if peak else 0.5))


def test_unknown_model_has_unknown_cost() -> None:
    assert PRICES.cost("deepseek", "nope", USAGE, MONDAY_PEAK) is None
    assert PRICES.cost("openai", "deepseek-flash", USAGE, MONDAY_PEAK) is None


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        PRICES.cost("deepseek", "deepseek-flash", USAGE, datetime(2026, 9, 28, 7))  # noqa: DTZ001  проверяем отказ на naive
