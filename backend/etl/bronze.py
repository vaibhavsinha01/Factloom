"""Bronze layer: raw page text from PDFs."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from backend.config import BRONZE_DIR, ensure_data_dirs
from backend.etl._io import write_parquet

logger = logging.getLogger("factloom.etl.bronze")


def write_bronze(
    document_id: str,
    document_version: int,
    pdf_path: str,
    pages: list[dict],
    content_hash: str,
) -> Path:
    ensure_data_dirs()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = BRONZE_DIR / document_id / f"v{document_version}"
    meta = {
        "document_id": document_id,
        "document_version": document_version,
        "content_hash": content_hash,
        "pdf_path": str(pdf_path),
        "num_pages": len(pages),
        "ingested_at": ts,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    rows = [
        {
            "document_id": document_id,
            "document_version": document_version,
            "page_no": p["page_no"],
            "text": p["text"],
            "content_hash": content_hash,
        }
        for p in pages
    ]
    write_parquet(out_dir / "pages.parquet", rows)
    logger.info("bronze written: %s (%d pages)", out_dir, len(pages))
    return out_dir
