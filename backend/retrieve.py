"""
Hybrid retrieval: metadata filtering → vector → lexical → reranking.

Removes artificial similarity boosting. Uses pgvector when available, with
FAISS as an in-process acceleration layer rebuilt from active facts.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter

from backend import embed as embed_mod
from backend.config import (
    RETRIEVAL_FINAL_K,
    RETRIEVAL_LEXICAL_K,
    RETRIEVAL_SIM_THRESHOLD,
    RETRIEVAL_VECTOR_K,
)
from backend.normalize import clean_entity

logger = logging.getLogger("factloom.retrieve")

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _fact_text(f: dict) -> str:
    parts = [
        f.get("entity") or "",
        f.get("metric") or "",
        f.get("value") or "",
        f.get("unit") or "",
        f.get("period") or "",
        f.get("scope") or "",
        f.get("geography") or "",
        f.get("reporting_basis") or "",
        f.get("quote") or "",
    ]
    return " ".join(parts)


def metadata_filter(fact: dict, candidates: list[dict]) -> list[dict]:
    """Hard filters: exclude self/same-doc; soft preference for entity/metric match.

    Returns the filtered pool (never artificially scored). Prefer same entity when
    available; if norm_metric is set, also keep same-metric facts even across entities
    only when entity is empty.
    """
    fact_id = fact.get("id")
    doc_id = fact.get("document_id")
    pool = [
        c for c in candidates
        if c.get("id") != fact_id and c.get("document_id") != doc_id
        and c.get("is_active", True)
    ]
    if not pool:
        return []

    fact_entity = clean_entity(fact.get("norm_entity") or fact.get("entity"))
    fact_metric = (fact.get("norm_metric") or "").strip().lower()

    if fact_entity:
        same_entity = [c for c in pool if clean_entity(c.get("norm_entity") or c.get("entity")) == fact_entity]
        if same_entity:
            pool = same_entity

    if fact_metric:
        same_metric = [c for c in pool if (c.get("norm_metric") or "").strip().lower() == fact_metric]
        # Prefer same metric but do not hard-require it (allows related metrics through vector/lexical)
        if same_metric:
            # Keep same-metric first by returning them; vector/lexical still rank within
            return same_metric + [c for c in pool if c not in same_metric]

    return pool


def vector_retrieve(fact: dict, pool: list[dict], k: int = RETRIEVAL_VECTOR_K) -> list[tuple[dict, float]]:
    """Semantic nearest neighbors via cosine similarity (no metric boosting)."""
    new_emb = embed_mod._safe_load_vector(fact.get("embedding"))
    if new_emb is None:
        return []

    scored: list[tuple[dict, float]] = []
    for cand in pool:
        cand_emb = embed_mod._safe_load_vector(cand.get("embedding"))
        if cand_emb is None:
            continue
        sim = embed_mod.cosine_sim(new_emb, cand_emb)
        if sim >= RETRIEVAL_SIM_THRESHOLD:
            scored.append((cand, sim))

    scored.sort(key=lambda x: -x[1])
    return scored[:k]


def lexical_retrieve(fact: dict, pool: list[dict], k: int = RETRIEVAL_LEXICAL_K) -> list[tuple[dict, float]]:
    """BM25-lite lexical retrieval over fact text fields."""
    query_tokens = _tokens(_fact_text(fact))
    if not query_tokens:
        return []

    df: Counter = Counter()
    docs_tokens: list[list[str]] = []
    for cand in pool:
        toks = _tokens(_fact_text(cand))
        docs_tokens.append(toks)
        for t in set(toks):
            df[t] += 1

    n = len(pool)
    avgdl = sum(len(t) for t in docs_tokens) / max(n, 1)
    k1, b = 1.2, 0.75
    q_counts = Counter(query_tokens)

    scored: list[tuple[dict, float]] = []
    for cand, toks in zip(pool, docs_tokens):
        if not toks:
            continue
        tf = Counter(toks)
        dl = len(toks)
        score = 0.0
        for term, qf in q_counts.items():
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * dl / max(avgdl, 1e-6))
            score += idf * (tf[term] * (k1 + 1) / denom) * qf
        if score > 0:
            scored.append((cand, score))

    scored.sort(key=lambda x: -x[1])
    return scored[:k]


def rerank(
    fact: dict,
    vector_hits: list[tuple[dict, float]],
    lexical_hits: list[tuple[dict, float]],
    k: int = RETRIEVAL_FINAL_K,
) -> list[dict]:
    """RRF (Reciprocal Rank Fusion) + structured-field bonus (not similarity boost)."""
    rrf_k = 60
    scores: dict[int, float] = {}
    by_id: dict[int, dict] = {}

    for rank, (cand, _) in enumerate(vector_hits):
        cid = cand["id"]
        by_id[cid] = cand
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

    for rank, (cand, _) in enumerate(lexical_hits):
        cid = cand["id"]
        by_id[cid] = cand
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

    fact_entity = clean_entity(fact.get("norm_entity") or fact.get("entity"))
    fact_metric = (fact.get("norm_metric") or "").strip().lower()
    fact_period = (fact.get("norm_period") or "").strip().lower()
    fact_scope = (fact.get("norm_scope") or "").strip().lower()

    for cid, cand in by_id.items():
        bonus = 0.0
        if fact_entity and clean_entity(cand.get("norm_entity") or cand.get("entity")) == fact_entity:
            bonus += 0.15
        if fact_metric and (cand.get("norm_metric") or "").strip().lower() == fact_metric:
            bonus += 0.20
        if fact_period and (cand.get("norm_period") or "").strip().lower() == fact_period:
            bonus += 0.10
        if fact_scope and fact_scope not in ("", "unspecified"):
            cs = (cand.get("norm_scope") or "").strip().lower()
            if cs == fact_scope:
                bonus += 0.05
        scores[cid] = scores.get(cid, 0.0) + bonus

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [by_id[cid] for cid, _ in ranked[:k] if cid in by_id]


def retrieve_candidates(
    fact: dict,
    all_facts: list[dict],
    k: int | None = None,
    threshold: float | None = None,  # kept for API compat; vector threshold from config
) -> list[dict]:
    """Full hybrid pipeline for cross-document candidate retrieval."""
    final_k = k or RETRIEVAL_FINAL_K
    fact_id = fact.get("id")

    pool = metadata_filter(fact, all_facts)
    if not pool:
        logger.info("fact_id=%s: no cross-document candidates after metadata filter", fact_id)
        return []

    # If metadata already narrowed to a small same-entity+metric set, still run hybrid
    vec = vector_retrieve(fact, pool, k=RETRIEVAL_VECTOR_K)
    lex = lexical_retrieve(fact, pool, k=RETRIEVAL_LEXICAL_K)

    if not vec and not lex:
        # Fall back to metadata-ordered pool (entity/metric preference already applied)
        return pool[:final_k]

    results = rerank(fact, vec, lex, k=final_k)
    validated = [r for r in results if r.get("id") is not None and r.get("id") != fact_id]
    logger.debug("fact_id=%s: hybrid retrieved %d candidates", fact_id, len(validated))
    return validated
