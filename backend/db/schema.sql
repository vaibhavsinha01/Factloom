-- FactLoom PostgreSQL + pgvector schema
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    num_pages       INTEGER,
    content_hash    TEXT NOT NULL,
    document_version INTEGER NOT NULL DEFAULT 1,
    s3_uri          TEXT,
    status          TEXT DEFAULT 'pending',
    error           TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_content_hash
    ON documents (content_hash);

CREATE TABLE IF NOT EXISTS processing_runs (
    id                  SERIAL PRIMARY KEY,
    document_id         TEXT NOT NULL REFERENCES documents(id),
    document_version    INTEGER NOT NULL,
    content_hash        TEXT NOT NULL,
    pipeline_version    TEXT NOT NULL,
    model_prompt_version TEXT NOT NULL,
    status              TEXT DEFAULT 'running',
    error               TEXT,
    facts_extracted     INTEGER DEFAULT 0,
    started_at          TIMESTAMPTZ DEFAULT NOW(),
    finished_at         TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_runs_document ON processing_runs (document_id);

CREATE TABLE IF NOT EXISTS facts (
    id                  SERIAL PRIMARY KEY,
    document_id         TEXT NOT NULL REFERENCES documents(id),
    document_version    INTEGER NOT NULL DEFAULT 1,
    run_id              INTEGER REFERENCES processing_runs(id),
    chunk_id            TEXT,
    page_no             INTEGER NOT NULL,
    entity              TEXT NOT NULL,
    metric              TEXT NOT NULL,
    value               TEXT NOT NULL,
    unit                TEXT,
    period              TEXT,
    scope               TEXT,
    geography           TEXT,
    reporting_basis     TEXT DEFAULT 'actual',
    quote               TEXT NOT NULL,
    confidence          TEXT DEFAULT 'medium',
    numeric_confidence  REAL DEFAULT 0.6,
    is_reported_value   BOOLEAN,
    evidence_status     TEXT DEFAULT 'pending',
    evidence_error      TEXT,
    embedding           vector(384),
    embedding_status    TEXT DEFAULT 'pending',
    embedding_model     TEXT,
    embedding_error     TEXT,
    norm_entity         TEXT,
    norm_metric         TEXT,
    norm_value          REAL,
    norm_unit           TEXT,
    norm_period         TEXT,
    norm_scope          TEXT,
    norm_geography      TEXT,
    norm_currency       TEXT,
    is_active           BOOLEAN DEFAULT TRUE,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_facts_document ON facts (document_id);
CREATE INDEX IF NOT EXISTS idx_facts_active ON facts (is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_facts_norm_metric ON facts (norm_metric);
CREATE INDEX IF NOT EXISTS idx_facts_norm_entity ON facts (norm_entity);
CREATE INDEX IF NOT EXISTS idx_facts_entity_metric_period
    ON facts (norm_entity, norm_metric, norm_period);

-- HNSW index for approximate nearest neighbor (pgvector)
CREATE INDEX IF NOT EXISTS idx_facts_embedding_hnsw
    ON facts USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL AND is_active = TRUE;

CREATE TABLE IF NOT EXISTS relations (
    id              SERIAL PRIMARY KEY,
    run_id          INTEGER REFERENCES processing_runs(id),
    fact_a_id       INTEGER NOT NULL REFERENCES facts(id),
    fact_b_id       INTEGER NOT NULL REFERENCES facts(id),
    relation_type   TEXT NOT NULL,
    explanation     TEXT,
    confidence      REAL,
    reason          TEXT DEFAULT 'none',
    fact_a_evidence TEXT,
    fact_b_evidence TEXT,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (fact_a_id, fact_b_id)
);

CREATE INDEX IF NOT EXISTS idx_relations_fact_a ON relations (fact_a_id);
CREATE INDEX IF NOT EXISTS idx_relations_fact_b ON relations (fact_b_id);

CREATE TABLE IF NOT EXISTS page_texts (
    document_id     TEXT NOT NULL REFERENCES documents(id),
    document_version INTEGER NOT NULL DEFAULT 1,
    page_no         INTEGER NOT NULL,
    text            TEXT NOT NULL,
    PRIMARY KEY (document_id, document_version, page_no)
);

CREATE TABLE IF NOT EXISTS jobs (
    id              SERIAL PRIMARY KEY,
    job_type        TEXT NOT NULL,
    payload         JSONB NOT NULL,
    status          TEXT DEFAULT 'queued',
    error           TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ
);
