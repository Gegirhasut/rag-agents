"""Стоимость вызова LLM по таблице цен (configs/llm_prices.yaml, ARCHITECTURE §12).

Считаем сами, а не полагаемся на Langfuse: у DeepSeek тарифы peak/off-peak (off-peak вдвое
дешевле), а model definition в Langfuse цен по времени суток не знает. Готовый cost уходит
в Langfuse через cost_details и перекрывает его собственный расчёт.
"""

from datetime import UTC, datetime, time
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from rag_agents.llm.base import LLMUsage

_PER_TOKEN = 1_000_000


class ModelPrice(BaseModel):
    """Цены за 1M токенов по peak-тарифу."""

    input: float
    cached_input: float
    output: float
    off_peak_multiplier: float = 1.0
    peak_hours_utc: list[str] = Field(default_factory=list)  # ["01:00-04:00", …]
    peak_weekdays: list[int] = Field(default_factory=lambda: list(range(7)))

    def is_peak(self, at: datetime) -> bool:
        if not self.peak_hours_utc:
            return True
        if at.utcoffset() is None:
            raise ValueError("datetime must be tz-aware")
        at_utc = at.astimezone(UTC)
        if at_utc.weekday() not in self.peak_weekdays:
            return False
        t = at_utc.time()
        for window in self.peak_hours_utc:
            start, end = (time.fromisoformat(x) for x in window.split("-"))
            if start <= t < end:
                return True
        return False


class CostBreakdown(BaseModel):
    input: float
    input_cache_read: float
    output: float
    peak: bool

    @property
    def total(self) -> float:
        return self.input + self.input_cache_read + self.output

    def as_langfuse(self) -> dict[str, float]:
        """Ключи совпадают с usage_details генерации (input, input_cache_read, output)."""
        return {
            "input": self.input,
            "input_cache_read": self.input_cache_read,
            "output": self.output,
            "total": self.total,
        }


class PriceTable:
    def __init__(self, prices: dict[str, dict[str, ModelPrice]]) -> None:
        self._prices = prices

    @classmethod
    def load(cls, path: Path) -> "PriceTable":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            {
                provider: {model: ModelPrice.model_validate(p) for model, p in models.items()}
                for provider, models in raw.items()
            }
        )

    def cost(
        self, provider: str, model: str, usage: LLMUsage, at: datetime
    ) -> CostBreakdown | None:
        """None — цены модели нет в таблице (стоимость неизвестна, а не ноль)."""
        price = self._prices.get(provider, {}).get(model)
        if price is None:
            return None
        peak = price.is_peak(at)
        k = (1.0 if peak else price.off_peak_multiplier) / _PER_TOKEN
        fresh_input = max(usage.input_tokens - usage.cached_input_tokens, 0)
        # reasoning у DeepSeek уже входит в output_tokens — отдельно не тарифицируем
        return CostBreakdown(
            input=fresh_input * price.input * k,
            input_cache_read=usage.cached_input_tokens * price.cached_input * k,
            output=usage.output_tokens * price.output * k,
            peak=peak,
        )


@lru_cache
def load_prices(path: Path) -> PriceTable:
    return PriceTable.load(path)
