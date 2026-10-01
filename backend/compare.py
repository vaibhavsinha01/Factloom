"""
Relationship classification:
    hybrid retrieval → deterministic pre-check → LLM comparison → validate → store

Deterministic logic considers entity, metric, period, scope, geography, unit,
currency, actual vs estimate, and reporting basis. Different periods/scopes/
units/currencies/definitions must NOT automatically become contradictions.
"""
from __future__ import annotations

import logging
import os

from backend import llm, store, validate as val
from backend.models import Relation, RelationType, ReconcileReason
from backend.normalize import clean_entity, canonical_unit
from backend.retrieve import retrieve_candidates

logger = logging.getLogger("factloom.compare")

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "compare_facts.txt")
with open(PROMPT_PATH, encoding="utf-8") as f:
    COMPARE_PROMPT = f.read()

_CORROBORATE_TOLERANCE = 0.01

# Unit families that are reconcilable via conversion rather than contradiction
_CURRENCY_UNITS = {"inr", "rs", "usd", "$", "eur", "gbp"}
_SCALE_INR = {"inr", "inr_lakh", "inr_crore", "inr_million", "inr_billion"}
_SCALE_USD = {"usd", "usd_million", "usd_billion"}


def _norm_scope(fact: dict) -> str:
    s = (fact.get("norm_scope") or fact.get("scope") or "").strip().lower()
    return "" if s in ("", "unspecified", "none", "null") else s


def _norm_geo(fact: dict) -> str:
    g = (fact.get("norm_geography") or fact.get("geography") or "").strip().lower()
    return "" if g in ("", "unspecified", "none", "null", "n/a") else g


def _reporting_basis(fact: dict) -> str:
    rb = (fact.get("reporting_basis") or "").strip().lower()
    if not rb:
        # Infer from is_reported_value when present
        irv = fact.get("is_reported_value")
        if irv is False or irv == 0:
            return "estimate"
        return "actual"
    if rb in ("reported",):
        return "actual"
    return rb


def _units_compatible(ua: str, ub: str) -> tuple[bool, bool]:
    """Return (compatible_same, reconcilable_via_unit_diff)."""
    ua = (ua or "unitless").lower()
    ub = (ub or "unitless").lower()
    if ua == ub:
        return True, False
    if ua == "unitless" or ub == "unitless":
        return True, False
    if {ua, ub} <= {"inr", "rs"}:
        return True, False
    if {ua, ub} <= {"usd", "$"}:
        return True, False
    if {ua, ub} <= {"percent", "pct", "%"}:
        return True, False
    # Different currency or scale → reconcilable, not contradict
    if (ua in _SCALE_INR and ub in _SCALE_USD) or (ua in _SCALE_USD and ub in _SCALE_INR):
        return False, True
    if (ua in _SCALE_INR and ub in _SCALE_INR and ua != ub) or (
        ua in _SCALE_USD and ub in _SCALE_USD and ua != ub
    ):
        return False, True
    if ua != ub:
        return False, True
    return False, False


def _deterministic_compare(fact_a: dict, fact_b: dict) -> Relation | None:
    """Classify when structurally obvious; return None to defer to LLM.

    NEVER returns CONTRADICTS for period/scope/unit/currency/geography/basis mismatches.
    Genuine contradictions require same entity+metric+period+scope+geo+basis+unit
    with values that differ beyond tolerance — still deferred to LLM unless extremely clear,
    except we DO return reconcilable/corroborates/unrelated/needs_review deterministically.
    """
    na, nb = fact_a.get("norm_metric"), fact_b.get("norm_metric")
    ua = fact_a.get("norm_unit") or canonical_unit(fact_a.get("unit"))
    ub = fact_b.get("norm_unit") or canonical_unit(fact_b.get("unit"))
    va, vb = fact_a.get("norm_value"), fact_b.get("norm_value")
    pa, pb = fact_a.get("norm_period"), fact_b.get("norm_period")
    ea = clean_entity(fact_a.get("norm_entity") or fact_a.get("entity"))
    eb = clean_entity(fact_b.get("norm_entity") or fact_b.get("entity"))

    fid_a = fact_a.get("id", 0)
    fid_b = fact_b.get("id", 0)

    def _rel(rtype, conf, reason, explanation):
        return Relation(
            fact_a_id=fid_a,
            fact_b_id=fid_b,
            relation_type=rtype,
            confidence=conf,
            reason=reason,
            explanation=explanation,
            fact_a_evidence=fact_a.get("quote") or "",
            fact_b_evidence=fact_b.get("quote") or "",
        )

    # Different entities → unrelated (unless one empty)
    if ea and eb and ea != eb:
        return _rel(
            RelationType.UNRELATED, 0.9, ReconcileReason.NONE,
            f"Different entities: '{ea}' vs '{eb}'.",
        )

    # Different metrics → unrelated
    if na and nb and na != nb:
        return _rel(
            RelationType.UNRELATED, 0.85, ReconcileReason.NONE,
            f"Different metrics: '{na}' vs '{nb}'.",
        )

    if not all([na, ua, va is not None, vb is not None, ea, eb]):
        # Incomplete normalization — needs review rather than forced contradiction
        if ea and eb and ea == eb and na and nb and na == nb:
            return _rel(
                RelationType.NEEDS_REVIEW, 0.4, ReconcileReason.NONE,
                "Same entity/metric but incomplete numeric normalization; manual review needed.",
            )
        return None

    if ea != eb or na != nb:
        return None

    sa, sb = _norm_scope(fact_a), _norm_scope(fact_b)
    ga, gb = _norm_geo(fact_a), _norm_geo(fact_b)
    ba, bb = _reporting_basis(fact_a), _reporting_basis(fact_b)

    # Actual vs estimate / different reporting basis → reconcilable, never contradict
    if ba and bb and ba != bb:
        return _rel(
            RelationType.RECONCILABLE, 0.88, ReconcileReason.ACTUAL_VS_ESTIMATE,
            f"Same {ea} {na} but reporting basis differs: {ba} vs {bb}.",
        )

    # Geography mismatch
    if ga and gb and ga != gb:
        return _rel(
            RelationType.RECONCILABLE, 0.9, ReconcileReason.DIFFERENT_GEOGRAPHY,
            f"Same {ea} {na} but geography differs: {ga} vs {gb}.",
        )

    # Scope mismatch (both stated)
    if sa and sb and sa != sb:
        return _rel(
            RelationType.RECONCILABLE, 0.9, ReconcileReason.DIFFERENT_SCOPE,
            f"Same {ea} {na} but scope differs: {sa} vs {sb}.",
        )

    same_unit, unit_diff_reconcilable = _units_compatible(ua, ub)
    if unit_diff_reconcilable and not same_unit:
        a_inr = ua.startswith("inr") or ua in ("inr", "rs")
        b_inr = ub.startswith("inr") or ub in ("inr", "rs")
        a_usd = ua.startswith("usd") or ua in ("usd", "$")
        b_usd = ub.startswith("usd") or ub in ("usd", "$")
        reason = (
            ReconcileReason.DIFFERENT_CURRENCY
            if (a_inr and b_usd) or (a_usd and b_inr)
            else ReconcileReason.DIFFERENT_UNIT
        )
        return _rel(
            RelationType.RECONCILABLE, 0.9, reason,
            f"Same {ea} {na} but units/currencies differ: {ua} vs {ub}.",
        )

    if not same_unit:
        return None

    try:
        av, bv = float(va), float(vb)
    except (TypeError, ValueError):
        return _rel(
            RelationType.NEEDS_REVIEW, 0.4, ReconcileReason.NONE,
            "Non-numeric values could not be compared deterministically.",
        )

    if av == 0 and bv == 0:
        return None

    max_val = max(abs(av), abs(bv))
    rel_diff = abs(av - bv) / max_val if max_val > 0 else 0.0
    periods_match = (pa == pb) or (not pa and not pb)

    if not periods_match and pa and pb:
        return _rel(
            RelationType.RECONCILABLE, 0.9, ReconcileReason.DIFFERENT_PERIOD,
            f"Both report {ea} {na} for different periods: {pa} vs {pb}. Values: {av} vs {bv} {ua}.",
        )

    if rel_diff <= _CORROBORATE_TOLERANCE:
        return _rel(
            RelationType.CORROBORATES, 0.95, ReconcileReason.NONE,
            f"Both facts report {ea} {na} as {av} {ua}"
            + (f" for {pa}" if pa else "")
            + f" (relative difference: {rel_diff * 100:.2f}%).",
        )

    if periods_match and rel_diff > _CORROBORATE_TOLERANCE:
        # Same period/scope/unit/basis — could be restatement OR genuine contradiction.
        # Mild differences → updated_information; large differences → needs_review / LLM.
        if rel_diff < 0.15:
            return _rel(
                RelationType.RECONCILABLE, 0.75, ReconcileReason.UPDATED_INFORMATION,
                f"Both report {ea} {na} for {pa or 'same period'} but values differ slightly: "
                f"{av} vs {bv} {ua} ({rel_diff * 100:.1f}%). Likely restatement/update.",
            )
        # Large gap with identical dimensions — defer to LLM for contradict vs restatement
        return None

    return None


def compare_new_facts(new_facts: list[dict]):
    """For each newly stored fact, retrieve candidates and classify relationships."""
    if not new_facts:
        return

    all_facts = store.get_all_facts(active_only=True)
    det_count = 0
    llm_count = 0

    for fact in new_facts:
        if fact.get("evidence_status") == "unverifiable":
            continue
        candidates = retrieve_candidates(fact, all_facts, k=5)

        for cand in candidates:
            if cand.get("evidence_status") == "unverifiable":
                continue
            existing = store.get_relations_for_fact(fact["id"])
            if any({r["fact_a_id"], r["fact_b_id"]} == {fact["id"], cand["id"]} for r in existing):
                continue

            det_relation = _deterministic_compare(fact, cand)
            if det_relation is not None:
                det_count += 1
                llm.increment_deterministic()
                if det_relation.relation_type not in (RelationType.UNRELATED,):
                    store.add_relation(
                        fact["id"], cand["id"],
                        det_relation.relation_type.value, det_relation.explanation,
                        confidence=det_relation.confidence, reason=det_relation.reason.value,
                        fact_a_evidence=det_relation.fact_a_evidence,
                        fact_b_evidence=det_relation.fact_b_evidence,
                        run_id=fact.get("run_id"),
                    )
                logger.debug(
                    "deterministic: fact %s vs %s → %s",
                    fact["id"], cand["id"], det_relation.relation_type.value,
                )
                continue

            llm_count += 1
            prompt = COMPARE_PROMPT.format(
                doc_a=fact["document_id"], page_a=fact["page_no"],
                entity_a=fact["entity"], metric_a=fact["metric"], value_a=fact["value"],
                unit_a=fact["unit"] or "", period_a=fact["period"] or "unspecified",
                scope_a=fact["scope"] or "unspecified",
                geography_a=fact.get("geography") or "unspecified",
                basis_a=_reporting_basis(fact),
                quote_a=fact["quote"],
                doc_b=cand["document_id"], page_b=cand["page_no"],
                entity_b=cand["entity"], metric_b=cand["metric"], value_b=cand["value"],
                unit_b=cand["unit"] or "", period_b=cand["period"] or "unspecified",
                scope_b=cand["scope"] or "unspecified",
                geography_b=cand.get("geography") or "unspecified",
                basis_b=_reporting_basis(cand),
                quote_b=cand["quote"],
            )
            try:
                result = llm.generate_json(prompt, call_type="relation")
            except llm.LLMUnavailableError as e:
                logger.error("compare failed for fact %s vs %s: %s", fact["id"], cand["id"], e)
                store.add_relation(
                    fact["id"], cand["id"],
                    RelationType.NEEDS_REVIEW.value,
                    f"LLM unavailable; flagged for review. ({e})",
                    confidence=0.3, reason=ReconcileReason.NONE.value,
                    fact_a_evidence=fact.get("quote") or "",
                    fact_b_evidence=cand.get("quote") or "",
                    run_id=fact.get("run_id"),
                )
                continue

            if not isinstance(result, dict):
                logger.warning("compare: expected dict, got %s", type(result))
                continue

            result["fact_a_id"] = fact["id"]
            result["fact_b_id"] = cand["id"]
            relation = val.validate_relation(result)
            if relation is None:
                continue
            if relation.relation_type == RelationType.UNRELATED:
                continue

            # Safety: never persist contradicts when dimensions clearly differ
            if relation.relation_type == RelationType.CONTRADICTS:
                guard = _deterministic_compare(fact, cand)
                if guard is not None and guard.relation_type == RelationType.RECONCILABLE:
                    relation = guard

            store.add_relation(
                fact["id"], cand["id"], relation.relation_type.value, relation.explanation,
                confidence=relation.confidence, reason=relation.reason.value,
                fact_a_evidence=relation.fact_a_evidence, fact_b_evidence=relation.fact_b_evidence,
                run_id=fact.get("run_id"),
            )

    logger.info(
        "compare_new_facts: processed=%d  deterministic=%d  llm_calls=%d",
        len(new_facts), det_count, llm_count,
    )
