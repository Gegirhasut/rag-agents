from dataclasses import dataclass

from redis.asyncio import Redis

# Sliding window log: ZSET с отметками времени запросов в окне. Атомарно в Lua, время —
# из Redis (TIME), чтобы два процесса uvicorn с расходящимися часами считали одинаково.
_SCRIPT = """
local key = KEYS[1]
local window = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
local count = redis.call('ZCARD', key)
if count < limit then
  redis.call('ZADD', key, now, now .. '-' .. t[2] .. '-' .. count)
  redis.call('PEXPIRE', key, window)
  return {1, limit - count - 1, 0}
end
local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
return {0, 0, tonumber(oldest[2]) + window - now}
"""


@dataclass(frozen=True)
class Rule:
    name: str
    limit: int
    window_s: int


@dataclass(frozen=True)
class RateDecision:
    allowed: bool
    remaining: int
    retry_after_s: int


class RateLimitedError(Exception):
    def __init__(self, rule: Rule, retry_after_s: int) -> None:
        super().__init__(f"rate limit {rule.name}: {rule.limit}/{rule.window_s}s")
        self.rule = rule
        self.retry_after_s = retry_after_s


class RateLimiter:
    """Лимиты FR-6.3: вопросы на пользователя, загрузки на пользователя, логин на IP."""

    def __init__(self, redis: Redis, *, enabled: bool = True) -> None:
        self.redis = redis
        self.enabled = enabled
        self._script = redis.register_script(_SCRIPT)

    async def hit(self, rule: Rule, subject: str) -> RateDecision:
        if not self.enabled:
            return RateDecision(allowed=True, remaining=rule.limit, retry_after_s=0)
        allowed, remaining, retry_ms = await self._script(
            keys=[f"rl:{rule.name}:{subject}"], args=[rule.window_s * 1000, rule.limit]
        )
        return RateDecision(
            allowed=bool(allowed),
            remaining=int(remaining),
            retry_after_s=max(1, -(-int(retry_ms) // 1000)) if not allowed else 0,
        )

    async def check(self, rule: Rule, subject: str) -> None:
        """Как hit, но превышение — исключение (роуты превращают его в 429)."""
        decision = await self.hit(rule, subject)
        if not decision.allowed:
            raise RateLimitedError(rule, decision.retry_after_s)
