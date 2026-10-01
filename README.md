# SuperJoin Financial Intelligence (FactLoom)

A fact knowledge layer for financial/economic PDFs: upload documents, extract grounded,
page-cited facts, and see how facts across documents corroborate, contradict, reconcile,
or need review — with confidence and evidence for every call.

## What changed in v2

- Hybrid retrieval: metadata → vector → lexical → RRF rerank (no artificial similarity boost)
- Stronger contradiction logic (period / scope / geography / unit / currency / actual-vs-estimate)
- Evidence grounding: quote must exist on the claimed page (no silent page rewrite)
- Primary store: PostgreSQL + pgvector (SQLite fallback for local/tests)
- MinIO/S3 for raw PDFs; Bronze → Silver → Gold Parquet lake
- Document versioning + content-hash idempotency (reprocess keeps history)
- Grounded RAG (`POST /ask`) that blocks unsupported numbers
- Async upload jobs via Kafka (sync fallback if Kafka is down)

---

## 1. Prerequisites

- Python 3.11+ (3.12/3.13 OK)
- Docker Desktop (for PostgreSQL/pgvector, MinIO, Kafka)
- Groq and/or Gemini API keys

---

## 2. Docker services

From the repo root:

```bash
docker compose up -d
```

This starts:

| Service    | Port(s)     | Purpose                |
|------------|-------------|------------------------|
| postgres   | 5432        | Primary DB + pgvector  |
| minio      | 9000 / 9001 | Raw PDF object store   |
| zookeeper  | 2181        | Kafka dependency       |
| kafka      | 9092        | Async upload jobs      |

Check health:

```bash
docker compose ps
```

---

## 3. `.env`

```bash
cp .env.example .env
```

Edit `.env` and set at least:

```env
GROQ_API_KEY=...
GEMINI_API_KEY=...   # optional fallback
STORE_BACKEND=postgres
DATABASE_URL=postgresql://factloom:factloom@localhost:5432/factloom
S3_ENABLED=true
KAFKA_ENABLED=true
KAFKA_FALLBACK_SYNC=true
```

Do not commit `.env`. Defaults in `.env.example` are local-dev only (not production secrets).

**Local/tests without Docker:**

```env
STORE_BACKEND=sqlite
S3_ENABLED=false
KAFKA_ENABLED=false
EMBEDDING_OFFLINE_FALLBACK=true
```

---

## 4. Database setup / migrations

Schema is applied automatically on API startup (`backend/db/schema.sql` via `store.init_db()`).

Manual apply (optional):

```bash
psql postgresql://factloom:factloom@localhost:5432/factloom -f backend/db/schema.sql
```

SQLite fallback DB path: `data/factloom.db`.

---

## 5. Backend startup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

Useful endpoints:

- `POST /upload?sync=true` — process inline (UI default)
- `POST /upload` — enqueue via Kafka (sync fallback)
- `GET /jobs/{id}` — job status
- `POST /ask` — grounded RAG (`{"question":"..."}`)
- `GET /metrics/history?entity=...&metric=...` — metric timeline
- `POST /reprocess` — re-run uploads (versioned; history retained)

---

## 6. Frontend startup

The SPA is served by FastAPI from `frontend/index.html` — no separate frontend server.

---

## 7. Tests

```bash
# Windows PowerShell
$env:STORE_BACKEND="sqlite"
$env:S3_ENABLED="false"
$env:KAFKA_ENABLED="false"
$env:EMBEDDING_OFFLINE_FALLBACK="true"
pytest -q
```

```bash
# macOS/Linux
STORE_BACKEND=sqlite S3_ENABLED=false KAFKA_ENABLED=false EMBEDDING_OFFLINE_FALLBACK=true pytest -q
```

Coverage includes: parsing, evidence validation, normalization, retrieval, contradiction
regression cases (period/scope/currency/actual-vs-estimate/restatement), idempotency, API.

---

## 8. Evaluation

Requires live LLM keys + network:

```bash
python -m backend.eval
python -m backend.eval --fresh-db   # wipe local sqlite DB first (sqlite mode)
```

Processes starter PDFs under `data/starter-datasets/` and prints relation summaries.

Optional smaller PDFs: `data/new_data/`.

---

## Pipeline (v2)

```
PDF
 → MinIO (raw) + Bronze Parquet (pages)
 → parse / candidate pages / chunking
 → LLM extract (Groq → Gemini)
 → evidence verify (document → page → quote → fact)
 → normalize → embed → PostgreSQL/pgvector + Silver/Gold
 → hybrid retrieve → deterministic compare → LLM compare
 → grounded RAG answers with citations
```

Spark is optional (`backend.etl.gold.spark_compact_gold`) for compacting many gold snapshots —
not required for normal operation.
