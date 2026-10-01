"""
Full per-document pipeline:

    parse → bronze → extract → evidence verify → validate → normalize → embed → silver/gold → store
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import fitz

from backend import embed as embed_mod
from backend import llm
from backend import normalize as norm_mod
from backend import store
from backend import validate as val
from backend.evidence import verify_fact_evidence
from backend.models import EmbeddingStatus
from backend.parse import parse_pdf_to_chunks, _extract_pages
from backend.storage import object_store
from backend.etl import bronze, silver, gold

logger = logging.getLogger("factloom.extract")

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "extract_facts.txt")
with open(PROMPT_PATH, encoding="utf-8") as f:
    EXTRACT_PROMPT = f.read()

_QUALITATIVE_TO_NUMERIC = {"high": 0.9, "medium": 0.6, "low": 0.35}


def _extract_chunk_facts(chunk: dict) -> list[dict]:
    """Call the LLM on one chunk and return raw fact dicts."""
    prompt = EXTRACT_PROMPT.format(
        document_id=chunk["document_id"],
        start_page=chunk["start_page"],
        end_page=chunk["end_page"],
        chunk_text=chunk["text"],
    )
    try:
        raw = llm.generate_json(prompt, call_type="extraction")
    except llm.LLMUnavailableError as e:
        logger.error("chunk %s extraction failed, all providers exhausted: %s", chunk["chunk_id"], e)
        return []

    if isinstance(raw, dict):
        for k in ("facts", "items", "data", "results"):
            if k in raw and isinstance(raw[k], list):
                raw = raw[k]
                break

    if not isinstance(raw, list):
        logger.warning("chunk %s: expected list, got %s", chunk["chunk_id"], type(raw))
        return []

    out = []
    for rf in raw:
        if not isinstance(rf, dict):
            continue
        page_no = rf.get("page_no")
        # Do NOT silently replace invalid pages with chunk start_page.
        # Leave as-is for evidence verification to flag/reject.
        if isinstance(page_no, str) and page_no.isdigit():
            page_no = int(page_no)
        if not isinstance(page_no, int):
            page_no = None

        conf_label = str(rf.get("confidence", "medium")).strip().lower()
        if conf_label not in _QUALITATIVE_TO_NUMERIC:
            conf_label = "medium"

        is_reported = rf.get("is_reported_value", True)
        reporting_basis = rf.get("reporting_basis")
        if not reporting_basis:
            reporting_basis = "actual" if is_reported else "estimate"

        out.append({
            "document_id": chunk["document_id"],
            "chunk_id": chunk["chunk_id"],
            "page_no": page_no if page_no is not None else chunk["start_page"],
            "_page_claimed": page_no,
            "_chunk_pages": list(chunk.get("page_map") or []),
            "entity": str(rf.get("entity", "")).strip(),
            "metric": str(rf.get("metric", "")).strip(),
            "value": str(rf.get("value", "")).strip(),
            "unit": rf.get("unit"),
            "period": rf.get("period"),
            "scope": rf.get("scope"),
            "geography": rf.get("geography"),
            "reporting_basis": reporting_basis,
            "quote": str(rf.get("quote", "")).strip(),
            "confidence": conf_label,
            "numeric_confidence": _QUALITATIVE_TO_NUMERIC[conf_label],
            "is_reported_value": is_reported,
        })
    return out


def process_pdf(pdf_path: str) -> list[dict]:
    """Run the full pipeline for one PDF. Returns stored facts (as dicts).

    Idempotent: identical content_hash skips reprocessing.
    Versioned: reprocessing soft-deactivates prior facts; history is retained.
    """
    pdf_path = str(pdf_path)
    doc_stem = Path(pdf_path).stem
    num_pages = fitz.open(pdf_path).page_count

    # Object store (best-effort)
    s3_uri = object_store.upload_pdf(pdf_path, key=f"raw/{doc_stem}/{Path(pdf_path).name}")

    reg = store.register_document(doc_stem, os.path.basename(pdf_path), num_pages, pdf_path, s3_uri)
    document_id = reg["document_id"]
    document_version = reg["document_version"]
    run_id = reg["run_id"]

    if reg.get("skip_processing"):
        logger.info(
            "idempotent skip: document_id=%s content_hash=%s already complete",
            document_id, reg["content_hash"],
        )
        store.set_document_status(document_id, "complete")
        return store.get_facts_by_document(document_id, active_only=True)

    # Bronze: raw page texts
    pages = _extract_pages(pdf_path)
    store.save_page_texts(document_id, document_version, pages)
    page_map = {p["page_no"]: p["text"] for p in pages}
    bronze.write_bronze(document_id, document_version, pdf_path, pages, reg["content_hash"])

    chunks_raw = parse_pdf_to_chunks(pdf_path)
    # Ensure document_id matches registered id (stem)
    for c in chunks_raw:
        c["document_id"] = document_id
    chunks = [c for c in chunks_raw if val.validate_chunk(c)]
    if not chunks:
        store.finish_run(run_id, 0, status="complete")
        store.set_document_status(document_id, "complete")
        return []

    stored_facts: list[dict] = []
    rejected = 0
    try:
        for chunk in chunks:
            raw_facts = _extract_chunk_facts(chunk)
            # Evidence grounding before schema validation softens page_no issues
            grounded = []
            for rf in raw_facts:
                claimed = rf.pop("_page_claimed", rf.get("page_no"))
                chunk_pages = rf.pop("_chunk_pages", chunk.get("page_map"))
                if claimed is not None:
                    rf["page_no"] = claimed
                ev = verify_fact_evidence(rf, page_map, chunk_page_map=chunk_pages)
                rf["page_no"] = ev["page_no"]
                rf["evidence_status"] = ev["evidence_status"]
                rf["evidence_error"] = ev["evidence_error"]
                if ev["evidence_status"] == "unverifiable":
                    rejected += 1
                    logger.info("rejected unverifiable fact: %s", rf.get("quote", "")[:80])
                    continue
                grounded.append(rf)

            valid_facts = val.validate_facts(grounded)

            for fact in valid_facts:
                normalized = norm_mod.normalize_fact(fact)
                norm_fields: dict = {}
                if normalized is not None:
                    validated_norm = val.validate_normalized([normalized.model_dump(exclude={"id"})])
                    if validated_norm:
                        n = validated_norm[0]
                        norm_fields = {
                            "norm_entity": n.entity,
                            "norm_metric": n.metric,
                            "norm_value": n.value,
                            "norm_unit": n.unit,
                            "norm_period": n.period,
                            "norm_scope": n.scope,
                            "norm_geography": n.geography,
                            "norm_currency": n.currency,
                        }

                embed_input = (
                    f"{fact.entity} | {fact.metric} | {fact.value} {fact.unit or ''} | "
                    f"{fact.period or ''} | {fact.scope or ''} | {fact.geography or ''} | "
                    f"{fact.reporting_basis or ''}"
                )
                emb_record = embed_mod.embed_text(embed_input)
                validated_emb = val.validate_embedding(emb_record.model_dump())

                embedding_fields = {
                    "embedding_status": (
                        validated_emb.status.value if validated_emb else EmbeddingStatus.FAILED.value
                    ),
                    "embedding_model": validated_emb.model if validated_emb else None,
                    "embedding_error": (
                        validated_emb.error
                        if validated_emb and validated_emb.status == EmbeddingStatus.FAILED
                        else emb_record.error
                    ),
                    "embedding_json": (
                        json.dumps(validated_emb.vector)
                        if (validated_emb and validated_emb.vector)
                        else None
                    ),
                }

                fact_dict = fact.model_dump(exclude={"id"})
                fact_dict.update(norm_fields)
                fact_dict.update(embedding_fields)
                fact_dict["document_version"] = document_version
                fact_dict["run_id"] = run_id

                fact_id = store.add_fact(fact_dict)
                fact_dict["id"] = fact_id
                fact_dict["embedding"] = embedding_fields["embedding_json"]
                stored_facts.append(fact_dict)

        # Silver / Gold lake layers (parquet)
        silver.write_silver(document_id, document_version, run_id, stored_facts)
        gold.write_gold_facts(stored_facts)

        store.finish_run(run_id, len(stored_facts), status="complete")
        store.set_document_status(document_id, "complete")
        logger.info(
            "document %s v%s complete: stored=%d rejected_unverifiable=%d",
            document_id, document_version, len(stored_facts), rejected,
        )
    except Exception as e:
        logger.exception("document %s processing failed", document_id)
        store.finish_run(run_id, len(stored_facts), status="failed", error=str(e))
        store.set_document_status(document_id, "failed", str(e))

    return stored_facts
