"""
Persistent store facade.

Primary: PostgreSQL + pgvector (STORE_BACKEND=postgres)
Fallback: SQLite (STORE_BACKEND=sqlite) for local tests without Docker.

Versioning: reprocessing never deletes historical facts/relations; prior rows are
marked is_active=FALSE and a new processing_run is recorded.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from backend.config import (
    DATABASE_URL,
    MODEL_PROMPT_VERSION,
    PIPELINE_VERSION,
    SQLITE_PATH,
    STORE_BACKEND,
    ensure_data_dirs,
)

logger = logging.getLogger("factloom.store")

# ---------- backend selection ----------

_USE_PG = STORE_BACKEND == "postgres"


def _set_backend(use_pg: bool) -> None:
    global _USE_PG
    _USE_PG = use_pg


def using_postgres() -> bool:
    return _USE_PG


def _content_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ===================== SQLite fallback =====================

_FACTS_NEW_COLUMNS = {
    "chunk_id": "TEXT",
    "is_reported_value": "INTEGER",
    "embedding_status": "TEXT DEFAULT 'pending'",
    "embedding_model": "TEXT",
    "embedding_error": "TEXT",
    "norm_metric": "TEXT",
    "norm_value": "REAL",
    "norm_unit": "TEXT",
    "norm_period": "TEXT",
    "norm_scope": "TEXT",
    "norm_entity": "TEXT",
    "norm_geography": "TEXT",
    "norm_currency": "TEXT",
    "numeric_confidence": "REAL",
    "geography": "TEXT",
    "reporting_basis": "TEXT DEFAULT 'actual'",
    "evidence_status": "TEXT DEFAULT 'pending'",
    "evidence_error": "TEXT",
    "document_version": "INTEGER DEFAULT 1",
    "run_id": "INTEGER",
    "is_active": "INTEGER DEFAULT 1",
    "content_hash": "TEXT",
}

_DOCS_NEW_COLUMNS = {
    "status": "TEXT DEFAULT 'pending'",
    "error": "TEXT",
    "content_hash": "TEXT",
    "document_version": "INTEGER DEFAULT 1",
    "s3_uri": "TEXT",
}

_RELATIONS_NEW_COLUMNS = {
    "confidence": "REAL",
    "reason": "TEXT DEFAULT 'none'",
    "fact_a_evidence": "TEXT",
    "fact_b_evidence": "TEXT",
    "run_id": "INTEGER",
    "is_active": "INTEGER DEFAULT 1",
}


def _sqlite_path() -> str:
    ensure_data_dirs()
    return str(SQLITE_PATH)


@contextmanager
def _sqlite_conn():
    conn = sqlite3.connect(_sqlite_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _migrate_columns(conn, table: str, columns: dict[str, str]):
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for col, coltype in columns.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")


def _init_sqlite():
    with _sqlite_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT,
                num_pages INTEGER,
                status TEXT DEFAULT 'pending',
                error TEXT,
                content_hash TEXT,
                document_version INTEGER DEFAULT 1,
                s3_uri TEXT
            );
            CREATE TABLE IF NOT EXISTS processing_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                document_version INTEGER,
                content_hash TEXT,
                pipeline_version TEXT,
                model_prompt_version TEXT,
                status TEXT DEFAULT 'running',
                error TEXT,
                facts_extracted INTEGER DEFAULT 0,
                started_at TEXT,
                finished_at TEXT
            );
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                page_no INTEGER,
                entity TEXT,
                metric TEXT,
                value TEXT,
                unit TEXT,
                period TEXT,
                scope TEXT,
                quote TEXT,
                confidence TEXT,
                embedding TEXT
            );
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_a_id INTEGER,
                fact_b_id INTEGER,
                relation_type TEXT,
                explanation TEXT
            );
            CREATE TABLE IF NOT EXISTS page_texts (
                document_id TEXT,
                document_version INTEGER DEFAULT 1,
                page_no INTEGER,
                text TEXT,
                PRIMARY KEY (document_id, document_version, page_no)
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type TEXT,
                payload TEXT,
                status TEXT DEFAULT 'queued',
                error TEXT,
                created_at TEXT,
                started_at TEXT,
                finished_at TEXT
            );
            """
        )
        _migrate_columns(conn, "facts", _FACTS_NEW_COLUMNS)
        _migrate_columns(conn, "documents", _DOCS_NEW_COLUMNS)
        _migrate_columns(conn, "relations", _RELATIONS_NEW_COLUMNS)


# ===================== Postgres =====================

def _init_postgres():
    from backend.db.postgres import init_schema
    init_schema()


def init_db():
    ensure_data_dirs()
    if STORE_BACKEND == "postgres":
        try:
            _init_postgres()
            _set_backend(True)
            logger.info("store backend=postgres url=%s", DATABASE_URL.split("@")[-1])
            return
        except Exception as e:
            logger.error("postgres init failed (%s); falling back to sqlite", e)
    _set_backend(False)
    _init_sqlite()
    logger.info("store backend=sqlite path=%s", _sqlite_path())


# Expose DB_PATH for eval --fresh-db compat
DB_PATH = str(SQLITE_PATH)


def _pg():
    from backend.db import postgres
    return postgres


# ===================== Documents / versioning =====================

def register_document(
    doc_id: str,
    filename: str,
    num_pages: int,
    pdf_path: str,
    s3_uri: str | None = None,
) -> dict:
    """Register or version a document. Identical content_hash → idempotent no-op run.

    Returns dict with keys: document_id, document_version, content_hash, run_id,
    skip_processing (True if identical hash already fully processed).
    """
    content_hash = _content_hash(pdf_path)

    if _USE_PG:
        try:
            return _register_document_pg(doc_id, filename, num_pages, content_hash, s3_uri)
        except Exception as e:
            logger.warning("pg register_document failed, sqlite fallback: %s", e)

    return _register_document_sqlite(doc_id, filename, num_pages, content_hash, s3_uri)


def _register_document_sqlite(doc_id, filename, num_pages, content_hash, s3_uri):
    with _sqlite_conn() as conn:
        # Idempotency: same content hash already complete?
        row = conn.execute(
            "SELECT * FROM documents WHERE content_hash=? AND status='complete'",
            (content_hash,),
        ).fetchone()
        if row:
            run_id = conn.execute(
                """INSERT INTO processing_runs
                (document_id, document_version, content_hash, pipeline_version,
                 model_prompt_version, status, facts_extracted, started_at, finished_at)
                VALUES (?,?,?,?,?,'skipped',0,?,?)""",
                (
                    row["id"], row["document_version"], content_hash,
                    PIPELINE_VERSION, MODEL_PROMPT_VERSION,
                    datetime.now(timezone.utc).isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            ).lastrowid
            return {
                "document_id": row["id"],
                "document_version": row["document_version"],
                "content_hash": content_hash,
                "run_id": run_id,
                "skip_processing": True,
            }

        existing = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        if existing:
            new_version = (existing["document_version"] or 1) + 1
            # Soft-deactivate prior facts/relations for this document
            fact_ids = [
                r[0] for r in conn.execute(
                    "SELECT id FROM facts WHERE document_id=? AND COALESCE(is_active,1)=1",
                    (doc_id,),
                ).fetchall()
            ]
            if fact_ids:
                conn.execute(
                    "UPDATE facts SET is_active=0 WHERE document_id=? AND COALESCE(is_active,1)=1",
                    (doc_id,),
                )
                placeholders = ",".join("?" for _ in fact_ids)
                conn.execute(
                    f"UPDATE relations SET is_active=0 WHERE fact_a_id IN ({placeholders}) OR fact_b_id IN ({placeholders})",
                    fact_ids + fact_ids,
                )
            conn.execute(
                """UPDATE documents SET filename=?, num_pages=?, status='processing', error=NULL,
                   content_hash=?, document_version=?, s3_uri=? WHERE id=?""",
                (filename, num_pages, content_hash, new_version, s3_uri, doc_id),
            )
            version = new_version
        else:
            conn.execute(
                """INSERT INTO documents (id, filename, num_pages, status, content_hash, document_version, s3_uri)
                   VALUES (?,?,?,'processing',?,1,?)""",
                (doc_id, filename, num_pages, content_hash, s3_uri),
            )
            version = 1

        run_id = conn.execute(
            """INSERT INTO processing_runs
            (document_id, document_version, content_hash, pipeline_version,
             model_prompt_version, status, started_at)
            VALUES (?,?,?,?,?,'running',?)""",
            (
                doc_id, version, content_hash, PIPELINE_VERSION, MODEL_PROMPT_VERSION,
                datetime.now(timezone.utc).isoformat(),
            ),
        ).lastrowid
        return {
            "document_id": doc_id,
            "document_version": version,
            "content_hash": content_hash,
            "run_id": run_id,
            "skip_processing": False,
        }


def _register_document_pg(doc_id, filename, num_pages, content_hash, s3_uri):
    with _pg().get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, document_version, status FROM documents WHERE content_hash=%s AND status='complete'",
                (content_hash,),
            )
            row = cur.fetchone()
            if row:
                cur.execute(
                    """INSERT INTO processing_runs
                    (document_id, document_version, content_hash, pipeline_version,
                     model_prompt_version, status, facts_extracted, finished_at)
                    VALUES (%s,%s,%s,%s,%s,'skipped',0,NOW()) RETURNING id""",
                    (row[0], row[1], content_hash, PIPELINE_VERSION, MODEL_PROMPT_VERSION),
                )
                run_id = cur.fetchone()[0]
                return {
                    "document_id": row[0],
                    "document_version": row[1],
                    "content_hash": content_hash,
                    "run_id": run_id,
                    "skip_processing": True,
                }

            cur.execute("SELECT document_version FROM documents WHERE id=%s", (doc_id,))
            existing = cur.fetchone()
            if existing:
                new_version = (existing[0] or 1) + 1
                cur.execute(
                    "UPDATE facts SET is_active=FALSE WHERE document_id=%s AND is_active=TRUE",
                    (doc_id,),
                )
                cur.execute(
                    """UPDATE relations SET is_active=FALSE
                       WHERE is_active=TRUE AND (
                         fact_a_id IN (SELECT id FROM facts WHERE document_id=%s) OR
                         fact_b_id IN (SELECT id FROM facts WHERE document_id=%s)
                       )""",
                    (doc_id, doc_id),
                )
                cur.execute(
                    """UPDATE documents SET filename=%s, num_pages=%s, status='processing', error=NULL,
                       content_hash=%s, document_version=%s, s3_uri=%s, updated_at=NOW() WHERE id=%s""",
                    (filename, num_pages, content_hash, new_version, s3_uri, doc_id),
                )
                version = new_version
            else:
                cur.execute(
                    """INSERT INTO documents (id, filename, num_pages, status, content_hash, document_version, s3_uri)
                       VALUES (%s,%s,%s,'processing',%s,1,%s)""",
                    (doc_id, filename, num_pages, content_hash, s3_uri),
                )
                version = 1

            cur.execute(
                """INSERT INTO processing_runs
                (document_id, document_version, content_hash, pipeline_version,
                 model_prompt_version, status)
                VALUES (%s,%s,%s,%s,%s,'running') RETURNING id""",
                (doc_id, version, content_hash, PIPELINE_VERSION, MODEL_PROMPT_VERSION),
            )
            run_id = cur.fetchone()[0]
            return {
                "document_id": doc_id,
                "document_version": version,
                "content_hash": content_hash,
                "run_id": run_id,
                "skip_processing": False,
            }


def finish_run(run_id: int, facts_extracted: int, status: str = "complete", error: str | None = None):
    if _USE_PG:
        try:
            with _pg().get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """UPDATE processing_runs SET status=%s, error=%s, facts_extracted=%s, finished_at=NOW()
                           WHERE id=%s""",
                        (status, error, facts_extracted, run_id),
                    )
            return
        except Exception as e:
            logger.warning("pg finish_run failed: %s", e)
    with _sqlite_conn() as conn:
        conn.execute(
            """UPDATE processing_runs SET status=?, error=?, facts_extracted=?, finished_at=?
               WHERE id=?""",
            (status, error, facts_extracted, datetime.now(timezone.utc).isoformat(), run_id),
        )


def set_document_status(doc_id: str, status: str, error: str | None = None):
    if _USE_PG:
        try:
            with _pg().get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE documents SET status=%s, error=%s, updated_at=NOW() WHERE id=%s",
                        (status, error, doc_id),
                    )
            return
        except Exception as e:
            logger.warning("pg set_document_status failed: %s", e)
    with _sqlite_conn() as conn:
        conn.execute("UPDATE documents SET status=?, error=? WHERE id=?", (status, error, doc_id))


def add_document(doc_id, filename, num_pages):
    """Legacy helper — prefer register_document."""
    if _USE_PG:
        try:
            with _pg().get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO documents (id, filename, num_pages, status, content_hash, document_version)
                           VALUES (%s,%s,%s,'processing','legacy',1)
                           ON CONFLICT (id) DO UPDATE SET filename=EXCLUDED.filename,
                           num_pages=EXCLUDED.num_pages, status='processing', updated_at=NOW()""",
                        (doc_id, filename, num_pages),
                    )
            return
        except Exception as e:
            logger.warning("pg add_document failed: %s", e)
    with _sqlite_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents (id, filename, num_pages, status) VALUES (?,?,?,?)",
            (doc_id, filename, num_pages, "processing"),
        )


def save_page_texts(document_id: str, document_version: int, pages: list[dict]):
    """pages: list of {page_no, text}"""
    if _USE_PG:
        try:
            with _pg().get_conn() as conn:
                with conn.cursor() as cur:
                    for p in pages:
                        cur.execute(
                            """INSERT INTO page_texts (document_id, document_version, page_no, text)
                               VALUES (%s,%s,%s,%s)
                               ON CONFLICT (document_id, document_version, page_no)
                               DO UPDATE SET text=EXCLUDED.text""",
                            (document_id, document_version, p["page_no"], p["text"]),
                        )
            return
        except Exception as e:
            logger.warning("pg save_page_texts failed: %s", e)
    with _sqlite_conn() as conn:
        for p in pages:
            conn.execute(
                """INSERT OR REPLACE INTO page_texts (document_id, document_version, page_no, text)
                   VALUES (?,?,?,?)""",
                (document_id, document_version, p["page_no"], p["text"]),
            )


def get_page_map(document_id: str, document_version: int | None = None) -> dict[int, str]:
    if _USE_PG:
        try:
            with _pg().get_conn() as conn:
                with conn.cursor() as cur:
                    if document_version is None:
                        cur.execute(
                            "SELECT page_no, text FROM page_texts WHERE document_id=%s",
                            (document_id,),
                        )
                    else:
                        cur.execute(
                            "SELECT page_no, text FROM page_texts WHERE document_id=%s AND document_version=%s",
                            (document_id, document_version),
                        )
                    return {r[0]: r[1] for r in cur.fetchall()}
        except Exception as e:
            logger.warning("pg get_page_map failed: %s", e)
    with _sqlite_conn() as conn:
        if document_version is None:
            rows = conn.execute(
                "SELECT page_no, text FROM page_texts WHERE document_id=?", (document_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT page_no, text FROM page_texts WHERE document_id=? AND document_version=?",
                (document_id, document_version),
            ).fetchall()
    return {r[0]: r[1] for r in rows}


def add_fact(fact: dict) -> int:
    embedding_json = fact.get("embedding_json")
    vector = None
    if embedding_json:
        try:
            vector = json.loads(embedding_json) if isinstance(embedding_json, str) else embedding_json
        except Exception:
            vector = None

    if _USE_PG:
        try:
            return _add_fact_pg(fact, vector)
        except Exception as e:
            logger.warning("pg add_fact failed: %s", e)

    with _sqlite_conn() as conn:
        cur = conn.execute(
            """INSERT INTO facts
            (document_id, document_version, run_id, chunk_id, page_no, entity, metric, value, unit,
             period, scope, geography, reporting_basis, quote, confidence, is_reported_value,
             embedding, embedding_status, embedding_model, embedding_error,
             norm_entity, norm_metric, norm_value, norm_unit, norm_period, norm_scope,
             norm_geography, norm_currency, numeric_confidence, evidence_status, evidence_error, is_active)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
            (
                fact["document_id"], fact.get("document_version", 1), fact.get("run_id"),
                fact.get("chunk_id"), fact["page_no"], fact["entity"], fact["metric"],
                fact["value"], fact.get("unit"), fact.get("period"), fact.get("scope"),
                fact.get("geography"), fact.get("reporting_basis", "actual"), fact["quote"],
                fact.get("confidence", "medium"), fact.get("is_reported_value"),
                embedding_json, fact.get("embedding_status", "pending"),
                fact.get("embedding_model"), fact.get("embedding_error"),
                fact.get("norm_entity"), fact.get("norm_metric"), fact.get("norm_value"),
                fact.get("norm_unit"), fact.get("norm_period"), fact.get("norm_scope"),
                fact.get("norm_geography"), fact.get("norm_currency"),
                fact.get("numeric_confidence", 0.6),
                fact.get("evidence_status", "pending"), fact.get("evidence_error"),
            ),
        )
        return cur.lastrowid


def _add_fact_pg(fact: dict, vector: list | None) -> int:
    with _pg().get_conn() as conn:
        with conn.cursor() as cur:
            emb_literal = None
            if vector:
                emb_literal = "[" + ",".join(str(float(x)) for x in vector) + "]"
            cur.execute(
                """INSERT INTO facts
                (document_id, document_version, run_id, chunk_id, page_no, entity, metric, value, unit,
                 period, scope, geography, reporting_basis, quote, confidence, is_reported_value,
                 embedding, embedding_status, embedding_model, embedding_error,
                 norm_entity, norm_metric, norm_value, norm_unit, norm_period, norm_scope,
                 norm_geography, norm_currency, numeric_confidence, evidence_status, evidence_error, is_active)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        %s::vector,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE)
                RETURNING id""",
                (
                    fact["document_id"], fact.get("document_version", 1), fact.get("run_id"),
                    fact.get("chunk_id"), fact["page_no"], fact["entity"], fact["metric"],
                    fact["value"], fact.get("unit"), fact.get("period"), fact.get("scope"),
                    fact.get("geography"), fact.get("reporting_basis", "actual"), fact["quote"],
                    fact.get("confidence", "medium"), fact.get("is_reported_value"),
                    emb_literal, fact.get("embedding_status", "pending"),
                    fact.get("embedding_model"), fact.get("embedding_error"),
                    fact.get("norm_entity"), fact.get("norm_metric"), fact.get("norm_value"),
                    fact.get("norm_unit"), fact.get("norm_period"), fact.get("norm_scope"),
                    fact.get("norm_geography"), fact.get("norm_currency"),
                    fact.get("numeric_confidence", 0.6),
                    fact.get("evidence_status", "pending"), fact.get("evidence_error"),
                ),
            )
            return cur.fetchone()[0]


def _row_to_fact(row) -> dict:
    if hasattr(row, "keys"):
        d = dict(row)
    else:
        d = row
    # Postgres vector → JSON string for embed helpers
    emb = d.get("embedding")
    if emb is not None and not isinstance(emb, str):
        try:
            if hasattr(emb, "tolist"):
                d["embedding"] = json.dumps(emb.tolist())
            elif isinstance(emb, (list, tuple)):
                d["embedding"] = json.dumps(list(emb))
            else:
                # pgvector may return string like '[0.1,0.2,...]'
                d["embedding"] = str(emb)
        except Exception:
            d["embedding"] = None
    return d


def get_all_facts(exclude_document_id: str | None = None, active_only: bool = True) -> list[dict]:
    if _USE_PG:
        try:
            with _pg().get_conn() as conn:
                with conn.cursor(cursor_factory=__import__("psycopg2.extras", fromlist=["RealDictCursor"]).RealDictCursor) as cur:
                    q = "SELECT * FROM facts"
                    clauses = []
                    params: list[Any] = []
                    if active_only:
                        clauses.append("is_active = TRUE")
                    if exclude_document_id:
                        clauses.append("document_id <> %s")
                        params.append(exclude_document_id)
                    if clauses:
                        q += " WHERE " + " AND ".join(clauses)
                    cur.execute(q, params)
                    return [_row_to_fact(dict(r)) for r in cur.fetchall()]
        except Exception as e:
            logger.warning("pg get_all_facts failed: %s", e)

    with _sqlite_conn() as conn:
        rows = conn.execute("SELECT * FROM facts").fetchall()
    facts = [dict(r) for r in rows]
    if active_only:
        facts = [f for f in facts if f.get("is_active", 1) in (1, True, None)]
    if exclude_document_id:
        facts = [f for f in facts if f["document_id"] != exclude_document_id]
    return facts


def get_facts_by_document(document_id: str, active_only: bool = True) -> list[dict]:
    if _USE_PG:
        try:
            with _pg().get_conn() as conn:
                with conn.cursor(cursor_factory=__import__("psycopg2.extras", fromlist=["RealDictCursor"]).RealDictCursor) as cur:
                    if active_only:
                        cur.execute(
                            "SELECT * FROM facts WHERE document_id=%s AND is_active=TRUE",
                            (document_id,),
                        )
                    else:
                        cur.execute("SELECT * FROM facts WHERE document_id=%s", (document_id,))
                    return [_row_to_fact(dict(r)) for r in cur.fetchall()]
        except Exception as e:
            logger.warning("pg get_facts_by_document failed: %s", e)

    with _sqlite_conn() as conn:
        rows = conn.execute("SELECT * FROM facts WHERE document_id=?", (document_id,)).fetchall()
    facts = [dict(r) for r in rows]
    if active_only:
        facts = [f for f in facts if f.get("is_active", 1) in (1, True, None)]
    return facts


def get_fact(fact_id: int) -> dict | None:
    if _USE_PG:
        try:
            with _pg().get_conn() as conn:
                with conn.cursor(cursor_factory=__import__("psycopg2.extras", fromlist=["RealDictCursor"]).RealDictCursor) as cur:
                    cur.execute("SELECT * FROM facts WHERE id=%s", (fact_id,))
                    row = cur.fetchone()
                    return _row_to_fact(dict(row)) if row else None
        except Exception as e:
            logger.warning("pg get_fact failed: %s", e)
    with _sqlite_conn() as conn:
        row = conn.execute("SELECT * FROM facts WHERE id=?", (fact_id,)).fetchone()
    return dict(row) if row else None


def add_relation(
    fact_a_id, fact_b_id, relation_type, explanation, confidence=None,
    reason="none", fact_a_evidence="", fact_b_evidence="", run_id=None,
):
    if _USE_PG:
        try:
            with _pg().get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO relations
                        (run_id, fact_a_id, fact_b_id, relation_type, explanation, confidence,
                         reason, fact_a_evidence, fact_b_evidence, is_active)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE)
                        ON CONFLICT (fact_a_id, fact_b_id) DO UPDATE SET
                          relation_type=EXCLUDED.relation_type,
                          explanation=EXCLUDED.explanation,
                          confidence=EXCLUDED.confidence,
                          reason=EXCLUDED.reason,
                          fact_a_evidence=EXCLUDED.fact_a_evidence,
                          fact_b_evidence=EXCLUDED.fact_b_evidence,
                          run_id=EXCLUDED.run_id,
                          is_active=TRUE
                        RETURNING id""",
                        (
                            run_id, fact_a_id, fact_b_id, relation_type, explanation,
                            confidence, reason, fact_a_evidence, fact_b_evidence,
                        ),
                    )
                    return cur.fetchone()[0]
        except Exception as e:
            logger.warning("pg add_relation failed: %s", e)

    with _sqlite_conn() as conn:
        cur = conn.execute(
            """INSERT INTO relations
            (run_id, fact_a_id, fact_b_id, relation_type, explanation, confidence,
             reason, fact_a_evidence, fact_b_evidence, is_active)
            VALUES (?,?,?,?,?,?,?,?,?,1)""",
            (
                run_id, fact_a_id, fact_b_id, relation_type, explanation,
                confidence, reason, fact_a_evidence, fact_b_evidence,
            ),
        )
        return cur.lastrowid


def get_relations_for_fact(fact_id: int, active_only: bool = True) -> list[dict]:
    if _USE_PG:
        try:
            with _pg().get_conn() as conn:
                with conn.cursor(cursor_factory=__import__("psycopg2.extras", fromlist=["RealDictCursor"]).RealDictCursor) as cur:
                    if active_only:
                        cur.execute(
                            "SELECT * FROM relations WHERE (fact_a_id=%s OR fact_b_id=%s) AND is_active=TRUE",
                            (fact_id, fact_id),
                        )
                    else:
                        cur.execute(
                            "SELECT * FROM relations WHERE fact_a_id=%s OR fact_b_id=%s",
                            (fact_id, fact_id),
                        )
                    return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.warning("pg get_relations_for_fact failed: %s", e)

    with _sqlite_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM relations WHERE fact_a_id=? OR fact_b_id=?", (fact_id, fact_id)
        ).fetchall()
    rels = [dict(r) for r in rows]
    if active_only:
        rels = [r for r in rels if r.get("is_active", 1) in (1, True, None)]
    return rels


def get_all_relations(active_only: bool = True) -> list[dict]:
    if _USE_PG:
        try:
            with _pg().get_conn() as conn:
                with conn.cursor(cursor_factory=__import__("psycopg2.extras", fromlist=["RealDictCursor"]).RealDictCursor) as cur:
                    if active_only:
                        cur.execute("SELECT * FROM relations WHERE is_active=TRUE")
                    else:
                        cur.execute("SELECT * FROM relations")
                    return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.warning("pg get_all_relations failed: %s", e)
    with _sqlite_conn() as conn:
        rows = conn.execute("SELECT * FROM relations").fetchall()
    rels = [dict(r) for r in rows]
    if active_only:
        rels = [r for r in rels if r.get("is_active", 1) in (1, True, None)]
    return rels


def list_documents() -> list[dict]:
    if _USE_PG:
        try:
            with _pg().get_conn() as conn:
                with conn.cursor(cursor_factory=__import__("psycopg2.extras", fromlist=["RealDictCursor"]).RealDictCursor) as cur:
                    cur.execute("SELECT * FROM documents ORDER BY updated_at DESC NULLS LAST")
                    return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.warning("pg list_documents failed: %s", e)
    with _sqlite_conn() as conn:
        rows = conn.execute("SELECT * FROM documents").fetchall()
    return [dict(r) for r in rows]


def delete_document_facts(document_id: str):
    """DEPRECATED for production — soft-deactivate instead. Kept for eval --fresh path."""
    logger.warning("delete_document_facts called for %s — prefer register_document soft-versioning", document_id)
    if _USE_PG:
        try:
            with _pg().get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE facts SET is_active=FALSE WHERE document_id=%s", (document_id,)
                    )
            return
        except Exception as e:
            logger.warning("pg delete_document_facts failed: %s", e)
    with _sqlite_conn() as conn:
        conn.execute(
            "UPDATE facts SET is_active=0 WHERE document_id=?", (document_id,)
        )


def get_metric_history(entity: str | None = None, metric: str | None = None) -> list[dict]:
    """Gold-layer style timeline: active facts for an entity/metric ordered by period."""
    facts = get_all_facts(active_only=True)
    from backend.normalize import clean_entity, canonical_metric

    out = []
    for f in facts:
        if entity and clean_entity(f.get("norm_entity") or f.get("entity")) != clean_entity(entity):
            continue
        if metric and (f.get("norm_metric") or canonical_metric(f.get("metric") or "")) != canonical_metric(metric):
            continue
        out.append({
            "id": f["id"],
            "document_id": f["document_id"],
            "entity": f.get("entity"),
            "metric": f.get("metric"),
            "norm_metric": f.get("norm_metric"),
            "value": f.get("value"),
            "norm_value": f.get("norm_value"),
            "unit": f.get("unit"),
            "norm_unit": f.get("norm_unit"),
            "period": f.get("period"),
            "norm_period": f.get("norm_period"),
            "scope": f.get("scope"),
            "geography": f.get("geography"),
            "reporting_basis": f.get("reporting_basis"),
            "page_no": f.get("page_no"),
            "quote": f.get("quote"),
            "confidence": f.get("confidence"),
            "numeric_confidence": f.get("numeric_confidence"),
        })
    out.sort(key=lambda x: (x.get("norm_period") or "", x.get("document_id") or ""))
    return out


def create_job(job_type: str, payload: dict) -> int:
    payload_s = json.dumps(payload)
    if _USE_PG:
        try:
            with _pg().get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO jobs (job_type, payload, status) VALUES (%s,%s::jsonb,'queued') RETURNING id",
                        (job_type, payload_s),
                    )
                    return cur.fetchone()[0]
        except Exception as e:
            logger.warning("pg create_job failed: %s", e)
    with _sqlite_conn() as conn:
        cur = conn.execute(
            "INSERT INTO jobs (job_type, payload, status, created_at) VALUES (?,?,?,?)",
            (job_type, payload_s, "queued", datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def update_job(job_id: int, status: str, error: str | None = None):
    now = datetime.now(timezone.utc).isoformat()
    if _USE_PG:
        try:
            with _pg().get_conn() as conn:
                with conn.cursor() as cur:
                    if status == "running":
                        cur.execute(
                            "UPDATE jobs SET status=%s, started_at=NOW(), error=%s WHERE id=%s",
                            (status, error, job_id),
                        )
                    else:
                        cur.execute(
                            "UPDATE jobs SET status=%s, finished_at=NOW(), error=%s WHERE id=%s",
                            (status, error, job_id),
                        )
            return
        except Exception as e:
            logger.warning("pg update_job failed: %s", e)
    with _sqlite_conn() as conn:
        if status == "running":
            conn.execute(
                "UPDATE jobs SET status=?, started_at=?, error=? WHERE id=?",
                (status, now, error, job_id),
            )
        else:
            conn.execute(
                "UPDATE jobs SET status=?, finished_at=?, error=? WHERE id=?",
                (status, now, error, job_id),
            )


def get_job(job_id: int) -> dict | None:
    if _USE_PG:
        try:
            with _pg().get_conn() as conn:
                with conn.cursor(cursor_factory=__import__("psycopg2.extras", fromlist=["RealDictCursor"]).RealDictCursor) as cur:
                    cur.execute("SELECT * FROM jobs WHERE id=%s", (job_id,))
                    row = cur.fetchone()
                    return dict(row) if row else None
        except Exception as e:
            logger.warning("pg get_job failed: %s", e)
    with _sqlite_conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(row) if row else None
