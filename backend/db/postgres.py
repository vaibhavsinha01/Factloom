"""PostgreSQL + pgvector connection helpers and schema migration."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import psycopg2
import psycopg2.extras

from backend.config import DATABASE_URL

logger = logging.getLogger("factloom.db.postgres")

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def get_connection():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    return conn


@contextmanager
def get_conn():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema():
    """Apply schema.sql idempotently."""
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    logger.info("PostgreSQL schema initialized")


def ping() -> bool:
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception as e:
        logger.warning("postgres ping failed: %s", e)
        return False
