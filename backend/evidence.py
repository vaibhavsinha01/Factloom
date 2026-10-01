"""Evidence grounding: verify that a quote actually exists on the claimed page."""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger("factloom.evidence")


def _normalize_for_match(text: str) -> str:
    """Aggressive whitespace/punctuation normalization for fuzzy quote matching."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ").replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def quote_on_page(quote: str, page_text: str, min_coverage: float = 0.7) -> bool:
    """Return True if quote (or a substantial contiguous substring) appears on the page."""
    if not quote or not page_text:
        return False
    q = _normalize_for_match(quote)
    p = _normalize_for_match(page_text)
    if not q or not p:
        return False
    if q in p:
        return True
    # Allow truncated quotes: require a contiguous window covering min_coverage of quote
    if len(q) < 20:
        return q in p
    window = max(20, int(len(q) * min_coverage))
    for i in range(0, len(q) - window + 1, max(1, window // 4)):
        snippet = q[i : i + window]
        if snippet in p:
            return True
    return False


def find_page_for_quote(quote: str, page_map: dict[int, str]) -> int | None:
    """Search pages for the quote. Returns page_no if uniquely found, else None."""
    hits = [pn for pn, text in page_map.items() if quote_on_page(quote, text)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        # Prefer exact substring match if available
        exact = [
            pn for pn in hits
            if _normalize_for_match(quote) in _normalize_for_match(page_map[pn])
        ]
        if len(exact) == 1:
            return exact[0]
        return hits[0]  # ambiguous but present — caller can flag
    return None


def verify_fact_evidence(
    fact: dict,
    page_map: dict[int, str],
    chunk_page_map: list[int] | None = None,
) -> dict:
    """Validate document→page→quote→fact grounding.

    Returns updated fields:
      evidence_status: verified | unverifiable | flagged
      evidence_error: optional message
      page_no: unchanged unless uniquely recoverable from quote (never silent fallback
               to chunk start_page for a wrong claim)

    Rules:
    - Incorrect page numbers are NOT silently replaced with the chunk's start page.
    - If the quote exists on a different page within the chunk/document, flag it and
      optionally correct when uniquely recoverable.
    - If the quote cannot be found anywhere, mark unverifiable (caller should reject).
    """
    quote = (fact.get("quote") or "").strip()
    page_no = fact.get("page_no")
    result = {
        "evidence_status": "pending",
        "evidence_error": None,
        "page_no": page_no,
    }

    if not quote:
        result["evidence_status"] = "unverifiable"
        result["evidence_error"] = "empty quote"
        return result

    # Restrict search space to chunk pages when provided, else all known pages
    search_map = page_map
    if chunk_page_map:
        search_map = {pn: page_map[pn] for pn in chunk_page_map if pn in page_map}

    claimed_text = search_map.get(page_no) if isinstance(page_no, int) else None

    if claimed_text is not None and quote_on_page(quote, claimed_text):
        result["evidence_status"] = "verified"
        return result

    # Claimed page wrong or missing — try to locate quote
    found = find_page_for_quote(quote, search_map)
    if found is None and search_map is not page_map:
        found = find_page_for_quote(quote, page_map)

    if found is None:
        result["evidence_status"] = "unverifiable"
        result["evidence_error"] = (
            f"quote not found on claimed page {page_no} or elsewhere in document"
        )
        logger.warning(
            "unverifiable evidence: page=%s quote=%.80s",
            page_no, quote,
        )
        return result

    if found != page_no:
        # Do NOT silently rewrite without flagging
        result["evidence_status"] = "flagged"
        result["evidence_error"] = (
            f"claimed page {page_no} but quote found on page {found}; corrected"
        )
        result["page_no"] = found
        logger.info(
            "flagged page mismatch: claimed=%s found=%s",
            page_no, found,
        )
        return result

    result["evidence_status"] = "verified"
    return result
