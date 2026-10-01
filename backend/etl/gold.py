"""Gold layer: analytics-ready fact snapshots + optional Spark compaction."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from backend.config import GOLD_DIR, ensure_data_dirs
from backend.etl._io import write_parquet

logger = logging.getLogger("factloom.etl.gold")


def write_gold_facts(facts: list[dict]) -> Path:
    ensure_data_dirs()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = GOLD_DIR / "facts" / f"snapshot_{ts}.parquet"
    rows = [
        {
            "id": f.get("id"),
            "document_id": f.get("document_id"),
            "norm_entity": f.get("norm_entity") or f.get("entity"),
            "norm_metric": f.get("norm_metric"),
            "norm_value": f.get("norm_value"),
            "norm_unit": f.get("norm_unit"),
            "norm_period": f.get("norm_period"),
            "norm_scope": f.get("norm_scope"),
            "reporting_basis": f.get("reporting_basis"),
            "page_no": f.get("page_no"),
            "quote": f.get("quote"),
            "numeric_confidence": f.get("numeric_confidence"),
        }
        for f in facts
        if f.get("evidence_status") != "unverifiable"
    ]
    write_parquet(out, rows)
    return out


def spark_compact_gold(input_glob: str | None = None) -> None:
    """Optional Spark compaction when many gold snapshots accumulate."""
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        logger.info("pyspark not installed — skipping gold compaction")
        return

    ensure_data_dirs()
    spark = SparkSession.builder.appName("factloom-gold-compact").master("local[*]").getOrCreate()
    path = input_glob or str(GOLD_DIR / "facts" / "*.parquet")
    try:
        df = spark.read.parquet(path)
        out = str(GOLD_DIR / "facts" / "compacted")
        df.coalesce(1).write.mode("overwrite").parquet(out)
        logger.info("spark compacted gold -> %s", out)
    finally:
        spark.stop()
