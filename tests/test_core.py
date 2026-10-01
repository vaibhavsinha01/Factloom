"""
Unit / regression tests for FactLoom upgrade.

Run with SQLite backend (no Docker required):
    set STORE_BACKEND=sqlite
    set S3_ENABLED=false
    set KAFKA_ENABLED=false
    set EMBEDDING_OFFLINE_FALLBACK=true
    pytest -q
"""
from __future__ import annotations

import os

# Force local backends before importing application modules
os.environ["STORE_BACKEND"] = "sqlite"
os.environ["S3_ENABLED"] = "false"
os.environ["KAFKA_ENABLED"] = "false"
os.environ["KAFKA_FALLBACK_SYNC"] = "true"
os.environ["EMBEDDING_OFFLINE_FALLBACK"] = "true"

import pytest

from backend.evidence import quote_on_page, verify_fact_evidence
from backend.models import RelationType, ReconcileReason
from backend.normalize import (
    canonical_metric,
    canonical_period,
    canonical_unit,
    clean_entity,
    normalize_fact,
    parse_numeric_value,
)
from backend.models import Fact
from backend.compare import _deterministic_compare
from backend.retrieve import metadata_filter, lexical_retrieve, rerank
from backend.rag import ground_numbers
from backend.parse import _is_candidate_page, _clean_page_text, _pages_to_chunks


# ---------- normalize ----------

def test_clean_entity_aliases():
    assert clean_entity("Bajaj Finance Ltd.") == "bajaj finance"
    assert clean_entity("M&M") == "mahindra & mahindra"


def test_canonical_metric_and_unit():
    assert canonical_metric("Revenue from operations") == "revenue"
    assert canonical_unit("%") == "percent"
    assert canonical_unit("Rs. crore") == "inr_crore"
    assert canonical_period("FY'24") == "FY2024"
    assert parse_numeric_value("8,993.6") == 8993.6


def test_normalize_fact_fields():
    fact = Fact(
        document_id="d1", page_no=1, entity="Bajaj Finance Ltd",
        metric="Revenue", value="1,234.5", unit="INR crore",
        period="FY2024", scope="consolidated", geography="India",
        reporting_basis="actual", quote="Revenue was 1,234.5",
    )
    n = normalize_fact(fact)
    assert n is not None
    assert n.metric == "revenue"
    assert n.unit == "inr_crore"
    assert n.currency == "INR"
    assert n.geography == "india"
    assert n.period == "FY2024"


# ---------- evidence ----------

def test_quote_on_page_exact():
    page = "Revenue for FY2024 was Rs. 1,234 crore according to management."
    assert quote_on_page("Revenue for FY2024 was Rs. 1,234 crore", page)


def test_verify_evidence_rejects_missing_quote():
    page_map = {1: "No numbers here.", 2: "Revenue was 100."}
    fact = {"page_no": 1, "quote": "EBITDA margin expanded to 42%"}
    ev = verify_fact_evidence(fact, page_map)
    assert ev["evidence_status"] == "unverifiable"


def test_verify_evidence_flags_wrong_page_not_silent():
    page_map = {1: "Intro text only.", 2: "Revenue was Rs. 100 crore in FY24."}
    fact = {"page_no": 1, "quote": "Revenue was Rs. 100 crore in FY24"}
    ev = verify_fact_evidence(fact, page_map, chunk_page_map=[1, 2])
    assert ev["evidence_status"] == "flagged"
    assert ev["page_no"] == 2
    assert "corrected" in (ev["evidence_error"] or "")


def test_verify_evidence_ok():
    page_map = {3: "Net profit stood at 50 crore."}
    fact = {"page_no": 3, "quote": "Net profit stood at 50 crore"}
    ev = verify_fact_evidence(fact, page_map)
    assert ev["evidence_status"] == "verified"


# ---------- parse ----------

def test_candidate_page_detection():
    assert _is_candidate_page("Revenue grew 15% in FY2024 to Rs. 1,000 crore")
    assert not _is_candidate_page("Table of contents\nChapter 1\nChapter 2")


def test_pages_to_chunks_preserve_page_markers():
    pages = [
        {"page_no": 1, "text": "Revenue was 10% in FY2023."},
        {"page_no": 2, "text": "Profit was 5% in FY2024."},
    ]
    chunks = _pages_to_chunks("doc", pages)
    assert chunks
    assert "[page 1]" in chunks[0]["text"] or any("[page 1]" in c["text"] for c in chunks)


# ---------- contradiction / comparison ----------

def _fact(**kwargs):
    base = {
        "id": kwargs.pop("id", 1),
        "document_id": kwargs.pop("document_id", "a"),
        "entity": "Acme",
        "norm_entity": "acme",
        "metric": "Revenue",
        "norm_metric": "revenue",
        "value": "100",
        "norm_value": 100.0,
        "unit": "%",
        "norm_unit": "percent",
        "period": "FY2024",
        "norm_period": "FY2024",
        "scope": "consolidated",
        "norm_scope": "consolidated",
        "geography": "India",
        "norm_geography": "india",
        "reporting_basis": "actual",
        "quote": "Revenue was 100",
    }
    base.update(kwargs)
    return base


def test_different_period_is_reconcilable_not_contradict():
    a = _fact(id=1, document_id="a")
    b = _fact(id=2, document_id="b", period="FY2023", norm_period="FY2023", norm_value=90.0, value="90")
    rel = _deterministic_compare(a, b)
    assert rel is not None
    assert rel.relation_type == RelationType.RECONCILABLE
    assert rel.reason == ReconcileReason.DIFFERENT_PERIOD


def test_different_scope_is_reconcilable():
    a = _fact(id=1)
    b = _fact(id=2, document_id="b", scope="standalone", norm_scope="standalone", norm_value=80.0)
    rel = _deterministic_compare(a, b)
    assert rel.relation_type == RelationType.RECONCILABLE
    assert rel.reason == ReconcileReason.DIFFERENT_SCOPE


def test_different_currency_is_reconcilable():
    a = _fact(id=1, norm_unit="inr_crore", unit="INR crore", norm_value=100.0)
    b = _fact(id=2, document_id="b", norm_unit="usd_million", unit="USD mn", norm_value=12.0)
    rel = _deterministic_compare(a, b)
    assert rel.relation_type == RelationType.RECONCILABLE
    assert rel.reason == ReconcileReason.DIFFERENT_CURRENCY


def test_actual_vs_estimate_is_reconcilable():
    a = _fact(id=1, reporting_basis="actual", norm_value=100.0)
    b = _fact(id=2, document_id="b", reporting_basis="estimate", norm_value=110.0)
    rel = _deterministic_compare(a, b)
    assert rel.relation_type == RelationType.RECONCILABLE
    assert rel.reason == ReconcileReason.ACTUAL_VS_ESTIMATE


def test_different_geography_is_reconcilable():
    a = _fact(id=1, geography="India", norm_geography="india")
    b = _fact(id=2, document_id="b", geography="Global", norm_geography="global", norm_value=200.0)
    rel = _deterministic_compare(a, b)
    assert rel.relation_type == RelationType.RECONCILABLE
    assert rel.reason == ReconcileReason.DIFFERENT_GEOGRAPHY


def test_same_everything_corroborates():
    a = _fact(id=1, norm_value=100.0)
    b = _fact(id=2, document_id="b", norm_value=100.5)
    rel = _deterministic_compare(a, b)
    assert rel.relation_type == RelationType.CORROBORATES


def test_restatement_small_diff_reconcilable():
    a = _fact(id=1, norm_value=100.0)
    b = _fact(id=2, document_id="b", norm_value=108.0)
    rel = _deterministic_compare(a, b)
    assert rel.relation_type == RelationType.RECONCILABLE
    assert rel.reason == ReconcileReason.UPDATED_INFORMATION


def test_genuine_large_gap_defers_to_llm():
    a = _fact(id=1, norm_value=100.0)
    b = _fact(id=2, document_id="b", norm_value=200.0)
    rel = _deterministic_compare(a, b)
    assert rel is None  # LLM / needs_review path


def test_different_entity_unrelated():
    a = _fact(id=1, entity="Acme", norm_entity="acme")
    b = _fact(id=2, document_id="b", entity="Beta", norm_entity="beta")
    rel = _deterministic_compare(a, b)
    assert rel.relation_type == RelationType.UNRELATED


# ---------- retrieval ----------

def test_metadata_filter_prefers_same_entity():
    fact = _fact(id=1, document_id="a")
    pool = [
        _fact(id=2, document_id="b", norm_entity="acme"),
        _fact(id=3, document_id="c", norm_entity="other", entity="Other"),
    ]
    filtered = metadata_filter(fact, pool)
    assert all(clean_entity(f.get("norm_entity")) == "acme" for f in filtered)


def test_no_artificial_boost_in_embed_top_similar():
    """Regression: metric match must not force similarity to 0.95."""
    from backend.embed import top_similar
    import json

    # Orthogonal-ish hashed vectors via offline fallback still may correlate;
    # instead assert the boost code path is gone by checking equal low sims stay low.
    v1 = [1.0] + [0.0] * 383
    v2 = [0.0, 1.0] + [0.0] * 382
    a = {"id": 1, "embedding": json.dumps(v1), "norm_metric": "revenue"}
    b = {"id": 2, "embedding": json.dumps(v2), "norm_metric": "revenue", "document_id": "x"}
    # threshold high — without boost, cosine of orthogonal vectors is 0
    results = top_similar(a, [b], k=5, threshold=0.5)
    assert results == []


# ---------- RAG grounding ----------

def test_ground_numbers_flags_unsupported():
    facts = [{"value": "100", "norm_value": 100, "quote": "Revenue was 100 crore"}]
    cleaned, removed = ground_numbers("Revenue was 100 and profit was 999", facts)
    assert "999" in removed or "[UNSUPPORTED:999]" in cleaned
    assert "100" not in removed


# ---------- idempotency ----------

def test_idempotent_identical_content(tmp_path):
    from backend import store as store_mod
    import hashlib

    store_mod.init_db()
    pdf = tmp_path / "sample.pdf"
    # Minimal valid-enough PDF bytes for hashing (not for parsing)
    pdf.write_bytes(b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n")

    r1 = store_mod.register_document("sample", "sample.pdf", 1, str(pdf))
    assert r1["skip_processing"] is False
    store_mod.set_document_status(r1["document_id"], "complete")
    store_mod.finish_run(r1["run_id"], 0, status="complete")

    r2 = store_mod.register_document("sample", "sample.pdf", 1, str(pdf))
    assert r2["skip_processing"] is True
    assert r2["content_hash"] == r1["content_hash"]


def test_reprocess_does_not_hard_delete(tmp_path):
    from backend import store as store_mod

    store_mod.init_db()
    pdf = tmp_path / "vdoc.pdf"
    pdf.write_bytes(b"%PDF-1.4 version1 content unique aaa\n%%EOF\n")
    r1 = store_mod.register_document("vdoc", "vdoc.pdf", 1, str(pdf))
    fid = store_mod.add_fact({
        "document_id": "vdoc",
        "document_version": 1,
        "run_id": r1["run_id"],
        "page_no": 1,
        "entity": "Acme",
        "metric": "Revenue",
        "value": "10",
        "quote": "Revenue 10",
        "evidence_status": "verified",
    })
    store_mod.set_document_status("vdoc", "complete")
    store_mod.finish_run(r1["run_id"], 1, "complete")

    pdf.write_bytes(b"%PDF-1.4 version2 content unique bbb\n%%EOF\n")
    r2 = store_mod.register_document("vdoc", "vdoc.pdf", 1, str(pdf))
    assert r2["skip_processing"] is False
    assert r2["document_version"] == 2

    old = store_mod.get_fact(fid)
    assert old is not None
    # Soft-deactivated
    assert old.get("is_active") in (0, False)


# ---------- API ----------

def test_api_documents_and_ask():
    from fastapi.testclient import TestClient
    # Re-init with sqlite
    from backend import store as store_mod
    store_mod.init_db()
    from backend.main import app

    client = TestClient(app)
    r = client.get("/documents")
    assert r.status_code == 200
    r = client.post("/ask", json={"question": "What was revenue?"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "cited_facts" in body
