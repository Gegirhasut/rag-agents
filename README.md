<p align="right"><b>English</b> · <a href="README.ru.md">Русский</a></p>

# RAG Agents — an experimental RAG playground for your own documents

**Create an agent, upload documents, ask questions. Answers come only from those materials, with citations pointing to the exact section of the source.**

A self-hosted RAG (Retrieval-Augmented Generation) system you can actually look inside: every step is visible — how a document is split into chunks, what the vectors look like, what retrieval found, how much an answer cost and how accurate it is by the metrics.

An agent is an isolated knowledge base with its own set of documents (txt, fb2, epub, pdf, docx). For each question it retrieves relevant fragments, streams the answer (SSE) and marks every claim with a `[n]` footnote linking to a specific section. If the materials don't contain the answer, the agent says so instead of making something up.

![Live schema: a question travelling through the services](docs/screenshots/live-schema.gif)

*The "Under the hood" page: a question goes browser → web → Ollama (embedding) → Qdrant (search) → DeepSeek (generation) → PostgreSQL and Langfuse. On the right is the event log streamed from Redis pub/sub.*

> The UI, code comments and design docs (`docs/`) are in Russian; commit messages and this README are in English.

## Why

A plain LLM chat answers "from memory": it mixes up details, invents facts and can't tell where something came from. Here the model sees only the retrieved fragments of your documents and has to cite them. Use cases:

- **corporate knowledge base:** policies, procedures, contracts, technical docs, wiki — one agent per team or project, with data isolated between agents;
- **product documentation assistant** via the JSON API: plug it into support, a bot or an internal portal;
- **RAG engineering playground:** every change to retrieval, chunking or prompts is measured on a golden dataset, not "by eye".

All tests, demos and evals use an open corpus — works by Leo Tolstoy (*War and Peace*, *A Confession*, in Russian): long texts with complex structure, several formats and encodings, and easy-to-verify questions for the golden dataset. Nothing in the system is specific to books.

It's a pet project and a portfolio piece: production practices (queues, idempotency, DLQ, data isolation, observability, evals) applied to a real task. Design decisions are documented in [ARCHITECTURE](docs/ARCHITECTURE.md).

## Features

- **5 formats:** txt (UTF-8, cp1251, koi8-r — encoding is detected automatically), fb2 and fb2.zip, epub, pdf with a text layer, docx; up to 100 MB per file.
- **Structural chunking:** the table of contents is taken from the file (TOC, styles, markup); chunks never cross section boundaries, and each one carries a path like "Volume 2 / Ch. XII" plus the page number for PDFs.
- **Queue-based ingestion pipeline:** parse → embed in fanned-out batches, live progress in the UI, idempotent tasks, retries for transient errors, DLQ with `make dlq-replay`, a sweeper for stuck documents.
- **Streamed answers with footnotes:** source cards with scores arrive first, then the text streams in; `[n]` links jump to the fragment.
- **Agent isolation:** every query to the database and Qdrant is filtered by `agent_id`; someone else's resource is a 404 (an integration test covers every endpoint).
- **JSON API** with keys, SSE streaming, OpenAPI and `application/problem+json` errors (RFC 9457); rate limits on questions, uploads and login.
- **Eval harness:** golden dataset, retrieval metrics (hit@k, recall@k, MRR), an LLM judge (faithfulness, answer relevancy, citation support), refusal and prompt-injection checks, run comparison with bootstrap confidence intervals.
- **Observability:** Langfuse Cloud traces for every question and every ingestion, cost with DeepSeek peak/off-peak pricing, Prometheus `/metrics`, JSON logs with `request_id`.
- **"Under the hood":** a live service schema, a Qdrant vector map (PCA), queues, workers and storage health — you can see what happens on every action.

## Screenshots

**Asking an agent** (demo corpus: *War and Peace*). The answer has footnotes to source sections; below it — model, time to first token, tokens (including reasoning), cost and a link to the trace.

![Chat with an agent](docs/screenshots/chat.png)

**Vector map.** Each dot is a chunk (1024 dimensions → 2 via PCA), colour is the document. ★ is the question vector, rings are the retrieved chunks with their scores.

![Qdrant vector map](docs/screenshots/vector-map.png)

**Quality (`/eval`).** Metrics of a golden-dataset run, a breakdown by question category, the "answerable / not answerable" boundary and a comparison of two runs with 95% CIs.

![Quality page](docs/screenshots/eval.png)

<table>
<tr>
<td width="50%"><b>Document:</b> table of contents from the file, chunks with token counts and per-stage timings<br><img src="docs/screenshots/document.png" alt="Document page"></td>
<td width="50%"><b>Analytics:</b> cost, latency, where the time goes, recent requests<br><img src="docs/screenshots/insights.png" alt="Analytics"></td>
</tr>
<tr>
<td><b>Agents</b><br><img src="docs/screenshots/agents.png" alt="Agent list"></td>
<td><b>JSON API</b> (OpenAPI, key-based auth)<br><img src="docs/screenshots/api-docs.png" alt="OpenAPI"></td>
</tr>
</table>

## How it works

```mermaid
flowchart LR
    subgraph Ingest["Ingestion (Celery)"]
        U[File upload] --> P["ingest.parse<br/>parse · clean · chunk by section"]
        P --> E["ingest.embed<br/>batches → bge-m3 (Ollama)"]
        E --> Q[(Qdrant<br/>agent_id in payload)]
        P --> PG[(PostgreSQL<br/>documents, chunks)]
    end
    subgraph Query["Answer (web, async)"]
        A[Question] --> QE[question embedding] --> S[top-k search<br/>in Qdrant by agent_id]
        S --> C[context + answer_v1 prompt] --> L[DeepSeek, streaming]
        L --> SSE[SSE: sources → tokens → done<br/>with n footnotes]
    end
    Q -.-> S
```

- **Web** is FastAPI with server-side rendering (Jinja2 + HTMX + SSE): no SPA, no node build.
- **Ingestion** runs in the background over RabbitMQ: messages carry IDs only; tasks are idempotent (claimed with a conditional `UPDATE … RETURNING`, deterministic uuid5 IDs for Qdrant points); errors are split into transient (retried) and permanent (immediately `failed`); anything else goes to a DLQ.
- **The RAG core is hand-written** (parsing, chunking, retrieval, prompts, citations) — no LangChain; LlamaIndex is planned only as a baseline to compare against in evals.
- Layering `web/api → services → repositories | rag | llm` is enforced by `import-linter`; ORM models never leave the repositories.

Full details (in Russian): [ARCHITECTURE](docs/ARCHITECTURE.md) (decisions, DB schema, contracts, memory budget), [SPEC](docs/SPEC.md) (requirements), [PLAN](docs/PLAN.md) (iterations). A visual overview for presentations: [docs/interview/index.html](docs/interview/index.html) (open it locally in a browser).

## Tech stack

| Layer | Technologies |
|---|---|
| Language & build | Python 3.12, [uv](https://docs.astral.sh/uv/), src layout, Docker (multi-stage), Docker Compose |
| Web & API | FastAPI, Jinja2, HTMX + `htmx-ext-sse`, Bootstrap 5, Pydantic v2, pydantic-settings |
| Data | PostgreSQL 16 (SQLAlchemy 2.0 async + asyncpg, Alembic), Redis 7 (sessions, rate limiting, pub/sub, cache) |
| Queues | Celery 5, RabbitMQ 3.13 (`ingest.parse`, `ingest.embed`, `maintenance`, `eval` queues and their DLQs), Celery beat |
| Search | Qdrant 1.19 (dense vectors, payload indexes on `agent_id`), `bge-m3` via Ollama (CPU), `tokenizers` |
| Parsing | lxml, selectolax (fb2, epub), PyMuPDF (pdf), python-docx, charset-normalizer, razdel |
| LLM | DeepSeek (OpenAI-compatible API) with streaming and reasoning effort |
| Observability | Langfuse Cloud (OpenTelemetry), structlog, prometheus_client |
| Code quality | ruff, mypy `--strict`, import-linter, pytest + pytest-asyncio, respx, polyfactory |

## System requirements

| | Minimum | Recommended | Notes |
|---|---|---|---|
| OS | Linux x86_64, macOS, or Windows with Docker Desktop (WSL2) | Linux | tested on Ubuntu 22.04 |
| CPU | 4 x86_64 cores **with AVX** | 6+ cores | without AVX, Ollama won't run in a container — use Ollama on the host instead (see below); no GPU needed |
| RAM | 8 GB | 12+ GB | container memory limits add up to ≈ 5.9 GB, 2.5 GB of it for Ollama with `bge-m3`; with Ollama on the host the stack needs ≈ 3.4 GB |
| Disk | 15 GB | 20+ GB | images ≈ 11 GB (Ollama 9.3 GB, the app 0.7 GB), the `bge-m3` model 1.2 GB, plus the index: ~330 MB per 9,000 chunks |
| Software | Docker Engine 24+ with Compose ≥ 2.24, GNU `make`, `git` | | for development — [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 itself) |
| Network | access to `api.deepseek.com` | | the first start downloads images and the model (~12 GB) |
| Keys | [DeepSeek API](https://platform.deepseek.com/) | Langfuse Cloud (free tier) | Langfuse is optional: without keys tracing is simply off |

Ports: the UI and API listen on `8080` (all interfaces). Qdrant Dashboard (`6333`), RabbitMQ UI (`15672`) and the debug admin UIs (`5555`, `8081`, `5540`) bind to `127.0.0.1` only; change that with `ADMIN_UI_BIND`.

Speed on CPU (6 vCPUs, no GPU): indexing one volume of *War and Peace* (~700 chunks) takes 1.5–4.5 minutes, mostly embedding; an answer takes 3–5 seconds, with the first token after ~2.5 seconds. A DeepSeek answer costs about $0.001.

## Quick start

```bash
git clone https://github.com/Gegirhasut/rag-agents.git
cd rag-agents
cp .env.example .env
```

Fill in `.env`:

- `DEEPSEEK_API_KEY` — your DeepSeek key;
- `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `RABBITMQ_DEFAULT_USER`, `RABBITMQ_DEFAULT_PASS` — any values (password: `openssl rand -hex 16`);
- `SEED_USER_EMAIL`, `SEED_USER_PASSWORD` — the admin account for the UI.

```bash
make up      # build the image and start the stack; the first run downloads bge-m3 (~1.2 GB)
make seed    # create the admin from SEED_USER_EMAIL / SEED_USER_PASSWORD
```

Open <http://localhost:8080>, sign in, click «Создать агента» (Create agent), upload a document and ask a question. The files in [`tests/fixtures/`](tests/fixtures) work for a first try — the same text in different formats. Migrations are applied by the one-off `migrate` service on every `make up`.

Without `make`: `docker compose up -d --build`, then `docker compose exec web rag-agents user create you@example.com --admin`.

### Where embeddings run

By default (`COMPOSE_PROFILES=local-ollama` in `.env`) Ollama with `bge-m3` runs in a container. If you already have Ollama on the host, remove `COMPOSE_PROFILES` and set `OLLAMA_BASE_URL` to it: `http://host.docker.internal:11434` with Docker Desktop, `http://10.0.2.2:11434` from a VirtualBox VM. Do the same on a CPU without AVX, where llama.cpp can't run in the container. On the host, install the model with `ollama pull bge-m3:567m`.

### Handy commands

```bash
make help          # all targets
make ps            # service status and memory
make logs s=web    # service logs (worker-ingest, worker-embed, beat, …)
make up-debug      # + Flower, pgweb, RedisInsight (linked from the "Under the hood" page)
make down          # stop (data stays in volumes)
```

Users: `docker compose exec web rag-agents user create EMAIL [--admin]` (asks for the password interactively); change a password with `rag-agents user set-password EMAIL`.

## JSON API

Issue a key in the UI on the «🔑 API-ключи» (API keys) page; it is shown once (only its hash is stored). Docs: <http://localhost:8080/api/docs>.

```bash
KEY=rag_…
curl -s -H "Authorization: Bearer $KEY" localhost:8080/api/v1/agents
curl -s -H "Authorization: Bearer $KEY" -F files=@handbook.pdf localhost:8080/api/v1/agents/$AGENT/documents
curl -N -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  localhost:8080/api/v1/agents/$AGENT/query -d '{"question": "What is the vacation policy?", "stream": true}'
```

With `"stream": true` you get SSE events `sources`, `token`…, `done`; without it — a single JSON (`chat_id`, `message_id`, `result`). Limits: 20 questions per minute per user, 30 uploads per hour, 5 login attempts per minute per IP; exceeding them returns `429` with `Retry-After`.

## Quality and observability

```bash
make eval AGENT=tolstoi DATASET=tolstoy CONFIG=dense   # golden-dataset run → /eval and reports/eval/
make eval-diff A=<run_id> B=<run_id>                   # compare two runs, 95% CI
make metrics                                           # the stack's Prometheus metrics
```

The golden dataset is [`eval/datasets/tolstoy.jsonl`](eval/datasets/tolstoy.jsonl): 59 questions over the demo corpus (facts, interpretation, multi-hop, out-of-corpus questions, prompt injection). Current baseline (dense retrieval, top-6): **hit@8 0.83 · recall@8 0.79 · MRR@10 0.57 · faithfulness 0.95 · refusal precision 0.93**.

Tracing: set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL` in `.env` (without keys tracing is off and everything else works). Check the connection with `make langfuse-check`. The «📊 Аналитика» (Analytics) page computes cost and latency from the app's own database and pulls each request's step tree from Langfuse. More in [ARCHITECTURE §14](docs/ARCHITECTURE.md).

The model and reasoning depth are set in `.env`: `LLM_MODEL` (`deepseek-flash`, `deepseek-v4-pro`) and `LLM_REASONING_EFFORT` (`low` / `high` / `max`). Reasoning models get a separate token allowance, `LLM_REASONING_BUDGET_TOKENS` (4,000 by default), on top of the answer budget, so long reasoning doesn't cut the answer short. Apply changes with `docker compose up -d web`.

## Development

```bash
uv python install 3.12
uv sync --all-extras
make lint          # ruff, ruff format --check, mypy --strict, import-linter
make test-unit     # unit tests, no docker
make test          # unit + integration on a separate compose.test.yaml (PG, Redis, Qdrant on tmpfs)
make smoke         # e2e against the running stack: agent → upload → done → question → stream
```

Code and architecture rules are in [CLAUDE.md](CLAUDE.md); step-by-step manual checks for each iteration are in [docs/MANUAL_TESTING.md](docs/MANUAL_TESTING.md) (both in Russian).

```
src/rag_agents/
├── core/          # Settings, DB (unit of work), Redis, logging, security
├── domain/        # Pydantic DTOs and enums — contracts between layers
├── models/        # ORM (repositories only)
├── repositories/  # PostgreSQL access, returns DTOs
├── services/      # use cases: agents, documents, ingest, answers, auth, eval
├── rag/           # parsing, cleaning, chunking, embeddings, index, retrieval, prompting
├── llm/           # LLM providers, pricing
├── web/  api/v1/  # HTML (HTMX) and JSON API
├── workers/       # Celery: queues, runtime, tasks
└── eval/          # runs, metrics, LLM judge, reports
```

## Status and roadmap

Iterations 1–4 of the [PLAN](docs/PLAN.md) are done: end-to-end skeleton, auth and API, production-grade ingestion, an eval harness with observability. Next:

- **5** — hybrid search (dense + BM25, RRF), the `bge-reranker-v2-m3` reranker, a calibrated refusal threshold, citation verification;
- **6** — the hand-written core vs LlamaIndex on the same dataset;
- **7** — `LLMRouter`: DeepSeek → Claude / OpenAI / Ollama fallback, circuit breaker, caching;
- **8** — multiple chats and an agentic mode (MVP); **9** — hardening; **10** — Streamlit and an MCP server.

## Licenses and privacy

| Component | License |
|---|---|
| `bge-m3` (embeddings) | MIT |
| `bge-reranker-v2-m3` (iteration 5) | Apache 2.0 |
| PyMuPDF (PDF parsing) | AGPL-3.0 — the only exception, replaceable with `pypdfium2` |
| Other libraries | MIT / BSD / Apache 2.0 |

Only models that allow commercial use are used. Non-commercial licenses (e.g. `jina-reranker-v2`) are excluded.

**Privacy.** Embeddings are computed locally, but the retrieved document fragments are sent together with the question to an external LLM (DeepSeek). For confidential documents use a provider you trust (`LLM_BASE_URL` accepts any OpenAI-compatible API, including a local one).

The project is licensed under [MIT](LICENSE).
