"""Проверка Langfuse Cloud: связь, цена модели, содержимое трейса.

    uv run python scripts/langfuse_check.py ping            # auth + тестовый трейс виден в API
    uv run python scripts/langfuse_check.py ensure-model    # цена LLM_MODEL для расчёта cost
    uv run python scripts/langfuse_check.py trace <id>      # дерево, токены и cost трейса

Ключи берутся из .env через Settings. Сами ключи скрипт не печатает.
"""

import argparse
import sys
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx

from rag_agents.core.config import Settings
from rag_agents.core.observability import LangfuseTracer, build_tracer

# DeepSeek, https://api-docs.deepseek.com/quick_start/pricing (проверено 2026-09-27), $ за 1M.
# Берём PEAK-цены: Langfuse не умеет тарифы по времени суток (off-peak вдвое дешевле),
# поэтому cost в Langfuse — оценка сверху. Точный расчёт — llm_prices.yaml (итерация 7).
PEAK_PRICES_PER_1M = {"input": 0.30, "input_cache_read": 0.006, "output": 1.20}
POLL_TIMEOUT_S = 60


def api(settings: Settings) -> httpx.Client:
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        sys.exit("FAIL: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY не заданы в .env")
    return httpx.Client(
        base_url=f"{settings.langfuse_base_url.rstrip('/')}/api/public",
        auth=(settings.langfuse_public_key, settings.langfuse_secret_key.get_secret_value()),
        timeout=15,
    )


def observations(client: httpx.Client, trace_id: str) -> list[dict[str, Any]]:
    """GET /api/public/traces/{id} для организаций с 16.09.2026 отключён (HTTP 410, legacy):
    трейс читаем как набор observations через v2 API."""
    r = client.get(
        "/v2/observations",
        params={
            "traceId": trace_id,
            "fields": "core,basic,time,model,usage,metrics,trace_context",
            "limit": 100,
        },
    )
    r.raise_for_status()
    data: list[dict[str, Any]] = r.json()["data"]
    return data


def wait_trace(
    client: httpx.Client, trace_id: str, min_observations: int = 1
) -> list[dict[str, Any]]:
    """Ingestion в Langfuse асинхронный: наблюдения появляются в API через секунды."""
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        obs = observations(client, trace_id)
        if len(obs) >= min_observations and all(o.get("endTime") for o in obs):
            return obs
        time.sleep(2)
    sys.exit(f"FAIL: трейс {trace_id} не появился за {POLL_TIMEOUT_S} с")


def ping(settings: Settings) -> None:
    with api(settings) as client:
        r = client.get("/projects")
        r.raise_for_status()
        project = r.json()["data"][0]
        print(f"auth ok: project «{project['name']}» ({settings.langfuse_base_url})")

        # В Settings app_env=dev → build_tracer отдаёт настоящий клиент
        tracer = build_tracer(settings)
        if not isinstance(tracer, LangfuseTracer):
            sys.exit("FAIL: трейсинг выключен (LANGFUSE_ENABLED=false или APP_ENV=test)")
        trace_id = tracer.trace_id_for(f"connectivity:{uuid4()}")
        root = tracer.start_trace(
            "connectivity-check",
            trace_id=trace_id,
            tags=["connectivity-check"],
            input={"sent_at": datetime.now(UTC).isoformat()},
        )
        with root.child("ping"):
            pass
        root.end(output="pong")
        tracer.shutdown()

        obs = wait_trace(client, trace_id, min_observations=2)
        names = sorted(o["name"] for o in obs)
        print(f"trace ok: {trace_id} observations={names}")
        print(
            f"UI: {settings.langfuse_base_url.rstrip('/')}"
            f"/project/{project['id']}/traces/{trace_id}"
        )


def ensure_model(settings: Settings) -> None:
    model = settings.llm_model
    prices = {k: v / 1_000_000 for k, v in PEAK_PRICES_PER_1M.items()}
    with api(settings) as client:
        existing = []
        page = 1
        while True:
            r = client.get("/models", params={"page": page, "limit": 100})
            r.raise_for_status()
            body = r.json()
            existing += [
                m
                for m in body["data"]
                if m["modelName"] == model and not m.get("isLangfuseManaged", False)
            ]
            if page >= body["meta"]["totalPages"]:
                break
            page += 1
        if existing:
            print(f"model definition уже есть: {model} (id={existing[0]['id']}), не трогаю")
            return
        r = client.post(
            "/models",
            json={
                "modelName": model,
                "matchPattern": f"(?i)^{model}$",
                "unit": "TOKENS",
                "pricingTiers": [
                    {
                        "name": "Standard (DeepSeek peak)",
                        "isDefault": True,
                        "priority": 0,
                        "conditions": [],
                        "prices": prices,
                    }
                ],
            },
        )
        if r.is_error:
            sys.exit(f"FAIL: HTTP {r.status_code}: {r.text[:300]}")
        print(
            f"model definition создан: {model} id={r.json()['id']} цены за 1M: {PEAK_PRICES_PER_1M}"
        )


def show_trace(settings: Settings, trace_id: str) -> None:
    with api(settings) as client:
        obs = sorted(wait_trace(client, trace_id), key=lambda o: o["startTime"])
        r = client.get("/v3/scores", params={"traceId": trace_id})
        scores = (
            [(sc["name"], sc["value"]) for sc in r.json().get("data", [])] if r.is_success else []
        )
    root = next(o for o in obs if o["isRootObservation"])
    cost = sum(o.get("totalCost") or 0 for o in obs)
    print(
        f"{root.get('traceName')}: session={root.get('sessionId')} user={root.get('userId')}"
        f" tags={root.get('tags')} latency={root.get('latency')}s cost=${cost:.6f}"
    )
    if scores:
        print(f"scores: {scores}")
    for o in obs:
        line = f"  {o['type']:<10} {o['name']:<24} {o.get('latency') or 0:>7.3f}s"
        if o["type"] == "GENERATION":
            line += (
                f" model={o.get('model')} usage={o.get('usageDetails')}"
                f" cost=${o.get('totalCost')} ttft={o.get('timeToFirstToken')}s"
            )
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ping")
    sub.add_parser("ensure-model")
    tr = sub.add_parser("trace")
    tr.add_argument("trace_id")
    args = parser.parse_args()
    settings = Settings(app_env="dev")
    match args.cmd:
        case "ping":
            ping(settings)
        case "ensure-model":
            ensure_model(settings)
        case "trace":
            show_trace(settings, args.trace_id)


if __name__ == "__main__":
    main()
