"""
Grounded RAG: question → metadata filters → hybrid retrieval → reranking →
facts → reasoning → cited answer.

Numerical claims in the answer must appear in retrieved facts; unsupported
numbers are stripped/flagged.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from backend import llm, store
from backend.normalize import clean_entity, canonical_metric
from backend.retrieve import lexical_retrieve, metadata_filter, rerank, vector_retrieve

logger = logging.getLogger("factloom.rag")

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")

RAG_PROMPT = """You are answering a financial question using ONLY the provided grounded facts.
Every number you mention MUST appear in one of the facts below. Cite facts as [fact_id].
If the facts are insufficient, say so clearly. Do not invent figures.

Question: {question}

Grounded facts:
{facts_block}

Return ONLY a JSON object:
{{
  "answer": "...",
  "cited_fact_ids": [1, 2],
  "unsupported_numbers_removed": []
}}
"""


def _apply_question_filters(question: str, facts: list[dict]) -> list[dict]:
    """Lightweight metadata filters inferred from the question text."""
    q = question.lower()
    pool = [f for f in facts if f.get("is_active", True) and f.get("evidence_status") != "unverifiable"]

    # Entity hint: match known entities present in facts
    entities = {clean_entity(f.get("norm_entity") or f.get("entity")) for f in pool}
    matched_entities = [e for e in entities if e and e in q]
    if matched_entities:
        pool = [
            f for f in pool
            if clean_entity(f.get("norm_entity") or f.get("entity")) in matched_entities
        ]

    # Metric aliases appearing in the question
    metric_hints = []
    for token in ("revenue", "profit", "ebitda", "gdp", "inflation", "eps", "margin", "aum", "npa"):
        if token in q:
            metric_hints.append(canonical_metric(token))
    if metric_hints:
        narrowed = [
            f for f in pool
            if (f.get("norm_metric") or "") in metric_hints
            or any(h in (f.get("metric") or "").lower() for h in metric_hints)
        ]
        if narrowed:
            pool = narrowed

    return pool


def _pseudo_query_fact(question: str) -> dict:
    """Build a synthetic fact dict so hybrid retrieval can score against real facts."""
    from backend import embed as embed_mod
    import json

    rec = embed_mod.embed_text(question)
    emb_json = json.dumps(rec.vector) if rec.vector else None
    return {
        "id": -1,
        "document_id": "__query__",
        "entity": "",
        "metric": question,
        "value": "",
        "unit": "",
        "period": "",
        "scope": "",
        "quote": question,
        "embedding": emb_json,
        "norm_metric": "",
        "norm_entity": "",
    }


def retrieve_for_question(question: str, k: int = 8) -> list[dict]:
    all_facts = store.get_all_facts(active_only=True)
    pool = _apply_question_filters(question, all_facts)
    if not pool:
        pool = [f for f in all_facts if f.get("evidence_status") != "unverifiable"]

    # metadata_filter expects a fact with document_id to exclude — use synthetic
    qfact = _pseudo_query_fact(question)
    # Don't use metadata_filter's same-doc exclusion against __query__
    vec = vector_retrieve(qfact, pool, k=20)
    lex = lexical_retrieve(qfact, pool, k=20)
    if not vec and not lex:
        return pool[:k]
    return rerank(qfact, vec, lex, k=k)


def _extract_numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", "") for m in _NUM_RE.finditer(text or "")}


def ground_numbers(answer: str, facts: list[dict]) -> tuple[str, list[str]]:
    """Remove or flag numbers in answer that do not appear in any fact value/quote."""
    allowed: set[str] = set()
    for f in facts:
        allowed |= _extract_numbers(str(f.get("value") or ""))
        allowed |= _extract_numbers(str(f.get("norm_value") or ""))
        allowed |= _extract_numbers(str(f.get("quote") or ""))

    removed = []
    parts = []
    last = 0
    for m in _NUM_RE.finditer(answer or ""):
        num = m.group(0).replace(",", "")
        # Allow years (19xx/20xx) and fact citation brackets nearby
        if re.fullmatch(r"(19|20)\d{2}", num):
            continue
        if num not in allowed and num.rstrip("%") not in allowed:
            removed.append(m.group(0))
            parts.append(answer[last:m.start()])
            parts.append(f"[UNSUPPORTED:{m.group(0)}]")
            last = m.end()
    parts.append(answer[last:] if answer else "")
    cleaned = "".join(parts)
    return cleaned, removed


def answer_question(question: str, k: int = 8) -> dict[str, Any]:
    facts = retrieve_for_question(question, k=k)
    if not facts:
        return {
            "answer": "No grounded facts available to answer this question.",
            "cited_facts": [],
            "cited_fact_ids": [],
            "unsupported_numbers_removed": [],
        }

    facts_block = "\n".join(
        f"[{f['id']}] {f.get('entity')} | {f.get('metric')} = {f.get('value')} {f.get('unit') or ''} "
        f"| period={f.get('period') or 'n/a'} | scope={f.get('scope') or 'n/a'} "
        f"| basis={f.get('reporting_basis') or 'actual'} | page={f.get('page_no')} "
        f"| quote=\"{f.get('quote')}\""
        for f in facts
    )
    prompt = RAG_PROMPT.format(question=question, facts_block=facts_block)

    try:
        result = llm.generate_json(prompt, call_type="relation")
    except llm.LLMUnavailableError as e:
        # Deterministic fallback: list top facts
        lines = [
            f"- [{f['id']}] {f.get('entity')} {f.get('metric')}={f.get('value')}{f.get('unit') or ''} "
            f"({f.get('period') or 'n/a'}, p{f.get('page_no')})"
            for f in facts[:5]
        ]
        return {
            "answer": "LLM unavailable; top grounded facts:\n" + "\n".join(lines),
            "cited_facts": facts,
            "cited_fact_ids": [f["id"] for f in facts],
            "unsupported_numbers_removed": [],
            "error": str(e),
        }

    if not isinstance(result, dict):
        result = {"answer": str(result), "cited_fact_ids": [f["id"] for f in facts]}

    answer = str(result.get("answer") or "")
    cited_ids = result.get("cited_fact_ids") or [f["id"] for f in facts]
    cited_set = set(int(x) for x in cited_ids if str(x).isdigit() or isinstance(x, int))
    cited_facts = [f for f in facts if f["id"] in cited_set] or facts

    cleaned, removed = ground_numbers(answer, cited_facts)
    return {
        "answer": cleaned,
        "cited_facts": [
            {
                "id": f["id"],
                "document_id": f.get("document_id"),
                "page_no": f.get("page_no"),
                "entity": f.get("entity"),
                "metric": f.get("metric"),
                "value": f.get("value"),
                "unit": f.get("unit"),
                "period": f.get("period"),
                "scope": f.get("scope"),
                "quote": f.get("quote"),
                "norm_value": f.get("norm_value"),
                "norm_unit": f.get("norm_unit"),
                "confidence": f.get("confidence"),
            }
            for f in cited_facts
        ],
        "cited_fact_ids": [f["id"] for f in cited_facts],
        "unsupported_numbers_removed": removed,
    }
