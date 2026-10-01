"""Silver layer: validated facts with evidence + normalization."""
from __future__ import annotations

import logging
from pathlib import Path

from backend.config import SILVER_DIR, ensure_data_dirs
from backend.etl._io import write_parquet

logger = logging.getLogger("factloom.etl.silver")


def write_silver(
    document_id: str,
    document_version: int,
    run_id: int,
    facts: list[dict],
) -> Path:
    ensure_data_dirs()
    out_dir = SILVER_DIR / document_id / f"v{document_version}" / f"run_{run_id}"
    rows = []
    for f in facts:
        rows.append({
            "id": f.get("id"),
            "document_id": f.get("document_id"),
            "document_version": document_version,
            "run_id": run_id,
            "page_no": f.get("page_no"),
            "entity": f.get("entity"),
            "metric": f.get("metric"),
            "value": f.get("value"),
            "unit": f.get("unit"),
            "period": f.get("period"),
            "scope": f.get("scope"),
            "geography": f.get("geography"),
            "reporting_basis": f.get("reporting_basis"),
            "quote": f.get("quote"),
            "evidence_status": f.get("evidence_status"),
            "norm_entity": f.get("norm_entity"),
            "norm_metric": f.get("norm_metric"),
            "norm_value": f.get("norm_value"),
            "norm_unit": f.get("norm_unit"),
            "norm_period": f.get("norm_period"),
            "norm_scope": f.get("norm_scope"),
            "confidence": f.get("confidence"),
            "numeric_confidence": f.get("numeric_confidence"),
        })
    write_parquet(out_dir / "facts.parquet", rows)
    logger.info("silver written: %s (%d facts)", out_dir, len(rows))
    return out_dir
