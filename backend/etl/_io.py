"""Shared lake I/O helpers."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("factloom.etl.io")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def write_parquet(path: Path, rows: list[dict]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        write_jsonl(path.with_suffix(".jsonl"), [])
        return False
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        clean = []
        for r in rows:
            clean.append({
                k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                for k, v in r.items()
            })
        table = pa.Table.from_pylist(clean)
        pq.write_table(table, path)
        return True
    except Exception as e:
        logger.info("parquet unavailable (%s); writing jsonl instead", e)
        write_jsonl(path.with_suffix(".jsonl"), rows)
        return False
