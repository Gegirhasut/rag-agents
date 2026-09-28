"""Замер ingest на живом стенде: время по документам и пиковая память контейнеров.

    uv run python scripts/ingest_bench.py FILE [FILE ...] [--agent "Имя"] [--kill-embed-at 30]

--kill-embed-at N: chaos-демо (PLAN, итерация 3) — `docker kill` worker-embed, когда эмбеддинг
документа пройдёт N %, затем `docker compose up -d worker-embed`. Проверяется, что документ
всё равно доезжает до done, а точек в Qdrant ровно chunks_total.
"""

import argparse
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from web_session import dotenv, login

CONTAINERS = ("worker-ingest", "worker-embed", "web", "postgres", "qdrant", "redis", "beat")
TERMINAL = {"done", "failed"}
COLLECTION = "chunks__bge_m3_567m__1024"
UNITS = {"KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0}
STEPS = ("parse_ms", "chunk_ms", "save_chunks_ms", "embed_ms", "upsert_ms")


class MemSampler(threading.Thread):
    """docker stats раз в 2 с: пиковая память каждого контейнера стенда."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak: dict[str, float] = {}
        self.stop = threading.Event()

    def run(self) -> None:
        cmd = ["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.MemUsage}}"]
        while not self.stop.is_set():
            out = subprocess.run(cmd, capture_output=True, text=True, check=False).stdout  # noqa: S603
            for line in out.splitlines():
                name, _, usage = line.partition(" ")
                m = re.match(r"([\d.]+)(\w+)", usage)
                if m and name.startswith("rag-agents-"):
                    key = name.removeprefix("rag-agents-").removesuffix("-1")
                    mib = float(m.group(1)) * UNITS.get(m.group(2), 0)
                    self.peak[key] = max(self.peak.get(key, 0), mib)
            self.stop.wait(2)


def api_client(base: str) -> httpx.Client:
    web = httpx.Client(base_url=base, timeout=120)
    login(web)
    r = web.post("/settings/api-keys", data={"name": "bench"}, headers={"HX-Request": "true"})
    m = re.search(r"(rag_[0-9a-f]{8}_[A-Za-z0-9_-]+)", r.text)
    if not m:
        sys.exit("FAIL: не выпустился API-ключ")
    return httpx.Client(
        base_url=base, headers={"Authorization": f"Bearer {m.group(1)}"}, timeout=300
    )


def embed_share(d: dict[str, Any]) -> float:
    return d["batches_done"] / d["batches_total"] if d["batches_total"] else 0.0


def kill_embed_worker() -> None:
    print(">>> docker kill rag-agents-worker-embed-1", flush=True)
    subprocess.run(["docker", "kill", "rag-agents-worker-embed-1"], check=True)  # noqa: S607
    time.sleep(5)
    subprocess.run(["docker", "compose", "up", "-d", "worker-embed"], check=True)  # noqa: S607


def duration(d: dict[str, Any]) -> str:
    if not (d.get("started_at") and d.get("finished_at")):
        return "—"
    delta = datetime.fromisoformat(d["finished_at"]) - datetime.fromisoformat(d["started_at"])
    return f"{delta.total_seconds():.0f}"


def steps(d: dict[str, Any]) -> str:
    t = (d.get("meta") or {}).get("timings", {})
    return " / ".join(f"{t.get(k, 0) / 1000:.1f}" for k in STEPS)


def points(qdrant: str, agent_id: str) -> int:
    body = {"exact": True, "filter": {"must": [{"key": "agent_id", "match": {"value": agent_id}}]}}
    r = httpx.post(f"{qdrant}/collections/{COLLECTION}/points/count", json=body, timeout=30)
    return int(r.json()["result"]["count"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--qdrant", default=f"http://{dotenv('ADMIN_UI_BIND') or '127.0.0.1'}:6333")
    ap.add_argument("--agent", default=f"Ingest bench {time.strftime('%H:%M')}")
    ap.add_argument("--kill-embed-at", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    api = api_client(args.base)
    agent = api.post("/api/v1/agents", json={"name": args.agent}).json()
    base = f"/api/v1/agents/{agent['id']}"
    print(f"агент: {agent['name']} ({agent['id']})")

    sampler = MemSampler()
    sampler.start()
    t0 = time.monotonic()
    for f in args.files:
        with f.open("rb") as fh:
            api.post(f"{base}/documents", files={"files": (f.name, fh)}).raise_for_status()
        print(f"загружен: {f.name} ({f.stat().st_size / 1e6:.1f} МБ)")

    killed = False
    docs: list[dict[str, Any]] = []
    while time.monotonic() - t0 < args.timeout:
        docs = api.get(f"{base}/documents").json()
        states = " | ".join(
            f"{d['filename'][:24]}: {d['status']} {d['progress']}%"
            + (f" {d['batches_done']}/{d['batches_total']}" if d["batches_total"] else "")
            for d in docs
        )
        print(f"[{time.monotonic() - t0:6.0f} с] {states}", flush=True)
        if (
            args.kill_embed_at is not None
            and not killed
            and any(embed_share(d) * 100 >= args.kill_embed_at for d in docs)
        ):
            kill_embed_worker()
            killed = True
        if docs and all(d["status"] in TERMINAL for d in docs):
            break
        time.sleep(5)
    sampler.stop.set()

    print("\nдокумент | статус | чанков | батчей | время, с | parse/chunk/save/embed/upsert, с")
    for d in docs:
        print(
            f"{d['filename']} | {d['status']} {d.get('error_code') or ''} | {d['chunks_total']} | "
            f"{d['batches_done']}/{d['batches_total']} | {duration(d)} | {steps(d)}"
        )
    print(f"\nвсего: {time.monotonic() - t0:.0f} с")
    peaks = ", ".join(f"{k} {v:.0f}" for k, v in sorted(sampler.peak.items()) if k in CONTAINERS)
    print(f"пиковая память, МиБ: {peaks}")
    expected = sum(d["chunks_total"] or 0 for d in docs if d["status"] == "done")
    print(f"точек в Qdrant: {points(args.qdrant, agent['id'])} · chunks_total (done): {expected}")


if __name__ == "__main__":
    main()
