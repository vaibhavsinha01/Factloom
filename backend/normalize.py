"""
Normalize raw extracted facts into a canonical representation so that semantically
equivalent facts (e.g. "revenue grew 15.2%" vs "revenue increased by 15.2 percent")
become comparable on structured fields rather than only on embedding similarity.

The original extracted value/unit/quote is always preserved (see NormalizedFact) —
normalization never destroys source information.
"""
import re
import logging
from backend.models import Fact, NormalizedFact

logger = logging.getLogger("factloom.normalize")

# --- entity canonicalization -------------------------------------------------
_LEGAL_SUFFIXES = re.compile(
    r"\b(ltd\.?|limited|corp\.?|corporation|inc\.?|incorporated|co\.?|company|pvt\.?|private)\b",
    re.IGNORECASE,
)

_ENTITY_ALIASES = {
    "m&m": "mahindra & mahindra",
    "m&m ltd": "mahindra & mahindra",
    "mahindra & mahindra ltd": "mahindra & mahindra",
    "mahindra and mahindra": "mahindra & mahindra",
    "bajaj finance ltd": "bajaj finance",
    "bajaj finance limited": "bajaj finance",
}


def clean_entity(entity: str | None) -> str:
    """Normalize company name to canonical lowercase form stripped of corporate suffixes."""
    if not entity:
        return ""
    s = entity.strip().lower()
    # Normalize punctuation and extra spaces
    s = re.sub(r"[,\.]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Check known aliases before and after suffix stripping
    if s in _ENTITY_ALIASES:
        return _ENTITY_ALIASES[s]
    cleaned = _LEGAL_SUFFIXES.sub("", s).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return _ENTITY_ALIASES.get(cleaned, cleaned or s)


# --- metric canonicalization -------------------------------------------------
# Maps common surface forms -> canonical snake_case metric name. Not exhaustive;
# unmatched metrics fall back to a slugified version of the original string, so
# nothing is dropped for lack of a mapping.
_METRIC_ALIASES = {
    # Revenue & Income
    "revenue": "revenue",
    "total revenue": "revenue",
    "revenue from operations": "revenue",
    "net sales": "revenue",
    "sales": "revenue",
    "revenue growth": "revenue_growth",
    # Profit & Loss
    "net loss": "net_loss",
    "net profit": "net_profit",
    "profit after tax": "net_profit",
    "pat": "net_profit",
    "pbt": "profit_before_tax",
    "profit before tax": "profit_before_tax",
    "ebitda": "ebitda",
    "ebitda margin": "ebitda_margin",
    "ebit": "ebit",
    # Per Share & Valuation
    "eps": "eps",
    "eps (inr)": "eps",
    "ttm eps": "eps",
    "ttm eps (inr)": "eps",
    "pe ratio": "pe_ratio",
    "pe ratio (ttm)": "pe_ratio",
    "ttm pe": "pe_ratio",
    "p/e": "pe_ratio",
    "p/e (x)": "pe_ratio",
    "roe": "roe",
    "roe (ttm)": "roe",
    "roe (%)": "roe",
    "return on equity": "roe",
    "roa": "roa",
    "return on assets": "roa",
    "div yield": "dividend_yield",
    "dividend yield": "dividend_yield",
    "dividend yield (ttm)": "dividend_yield",
    # Stock & Recommendation Data
    "cmp": "current_market_price",
    "current market price": "current_market_price",
    "market price": "current_market_price",
    "stock price": "current_market_price",
    "target price": "target_price",
    "target price (rs.)": "target_price",
    "target price (rs)": "target_price",
    "tp": "target_price",
    "12-month target price": "target_price",
    "12m target price": "target_price",
    "upside": "upside",
    "upside potential": "upside",
    "stop loss": "stop_loss",
    "stop loss (closing basis)": "stop_loss",
    "sl": "stop_loss",
    "market cap": "market_cap",
    "market cap (inr mn)": "market_cap_inr",
    "market cap ($ mn)": "market_cap_usd",
    "market cap (usd mn)": "market_cap_usd",
    "market capitalization": "market_cap",
    "mcap": "market_cap",
    "shares o/s": "shares_outstanding",
    "shares o/s (in mn)": "shares_outstanding",
    "shares outstanding": "shares_outstanding",
    "shares outstanding (mn)": "shares_outstanding",
    "avg. volume": "avg_volume",
    "avg. volume (3 month)": "avg_volume_3m",
    "avg volume": "avg_volume",
    "average volume": "avg_volume",
    "52-week high": "52_week_high",
    "52-week high (rs.)": "52_week_high",
    "52-week high (rs)": "52_week_high",
    "52-week range high": "52_week_high",
    "52-week range high (rs.)": "52_week_high",
    "52-week low": "52_week_low",
    "52-week low (rs.)": "52_week_low",
    "52-week low (rs)": "52_week_low",
    "52-week range low": "52_week_low",
    "52-week range low (rs.)": "52_week_low",
    # Shareholding Pattern
    "promoter shareholding": "promoter_shareholding",
    "promoters": "promoter_shareholding",
    "shareholding - promoters": "promoter_shareholding",
    "fii shareholding": "fii_shareholding",
    "fiis": "fii_shareholding",
    "fii": "fii_shareholding",
    "shareholding - fiis": "fii_shareholding",
    "institutional shareholding": "institutional_shareholding",
    "institutions": "institutional_shareholding",
    "shareholding - institutions": "institutional_shareholding",
    "other shareholding": "other_shareholding",
    "others": "other_shareholding",
    "others (incl. body corporate)": "other_shareholding",
    "shareholding - others (incl. body corporate)": "other_shareholding",
    # Performance
    "1-month performance": "performance_1m",
    "1m performance": "performance_1m",
    "6-month performance": "performance_6m",
    "6m performance": "performance_6m",
    "1-year performance": "performance_1y",
    "1yr performance": "performance_1y",
    # Banking / Financial
    "aum": "aum",
    "nim": "nim",
    "net interest margin": "nim",
    "gross npa": "gross_npa",
    "gnpa": "gross_npa",
    "npa": "npa",
    "gross npa ratio": "gross_npa_ratio",
    # Macro
    "gdp growth": "gdp_growth",
    "gdp growth rate": "gdp_growth",
    "real gdp growth": "gdp_growth",
    "inflation": "inflation_rate",
    "inflation rate": "inflation_rate",
    "cpi inflation": "inflation_rate",
    "repo rate": "repo_rate",
    "current account deficit": "current_account_deficit",
    "fiscal deficit": "fiscal_deficit",
    "foreign exchange reserves": "forex_reserves",
    "forex reserves": "forex_reserves",
}


def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def canonical_metric(metric: str) -> str:
    key = metric.strip().lower()
    return _METRIC_ALIASES.get(key, _slugify(metric))


# --- unit canonicalization ----------------------------------------------------
_UNIT_ALIASES = {
    "%": "percent", "percent": "percent", "percentage": "percent", "pct": "percent",
    "inr": "inr", "rs": "inr", "rs.": "inr", "rupees": "inr",
    "inr crore": "inr_crore", "rs. crore": "inr_crore", "rs crore": "inr_crore", "crore": "inr_crore", "crores": "inr_crore",
    "inr lakh": "inr_lakh", "lakh": "inr_lakh",
    "inr mn": "inr_million", "inr million": "inr_million", "rs mn": "inr_million", "rs. mn": "inr_million", "rs million": "inr_million",
    "inr billion": "inr_billion", "rs.764bn": "inr_billion",
    "usd": "usd", "$": "usd", "us$": "usd",
    "usd mn": "usd_million", "$ mn": "usd_million", "usd million": "usd_million", "usd billion": "usd_billion",
    "bps": "basis_points", "basis points": "basis_points",
    "days": "days", "x": "multiple", "times": "multiple",
}


def canonical_unit(unit: str | None) -> str:
    if not unit:
        return "unitless"
    key = unit.strip().lower()
    return _UNIT_ALIASES.get(key, _slugify(unit))


# --- period canonicalization ---------------------------------------------------
_FY_RE = re.compile(r"fy\s*'?(\d{2,4})", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_QTR_RE = re.compile(r"\bq([1-4])\b", re.IGNORECASE)


def canonical_period(period: str | None) -> str | None:
    if not period:
        return None
    p = period.strip()
    qtr_match = _QTR_RE.search(p)
    fy_match = _FY_RE.search(p)
    if fy_match:
        yr = fy_match.group(1)
        yr = ("20" + yr) if len(yr) == 2 else yr
        prefix = f"Q{qtr_match.group(1)} " if qtr_match else ""
        return f"{prefix}FY{yr}"
    year_match = _YEAR_RE.search(p)
    if year_match:
        return year_match.group(0)
    return p  # leave as-is if nothing recognizable; don't destroy information


# --- value parsing ---------------------------------------------------------
_NUM_RE = re.compile(r"-?[\d,]+\.?\d*")


def parse_numeric_value(value: str) -> float | None:
    """Extract a float from a value string like '8,993.6' or '15.2%'. Returns None if non-numeric
    (e.g. a status like 'resigned') — such facts are simply not normalized numerically."""
    cleaned = value.replace(",", "")
    match = _NUM_RE.search(cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


_GEO_ALIASES = {
    "india": "india",
    "indian": "india",
    "in": "india",
    "us": "united_states",
    "usa": "united_states",
    "u.s.": "united_states",
    "united states": "united_states",
    "global": "global",
    "worldwide": "global",
    "domestic": "domestic",
    "international": "international",
}


def canonical_geography(geo: str | None) -> str | None:
    if not geo:
        return None
    key = geo.strip().lower()
    return _GEO_ALIASES.get(key, _slugify(geo))


def infer_currency(unit: str | None) -> str | None:
    u = canonical_unit(unit)
    if u.startswith("inr") or u in ("inr", "rs"):
        return "INR"
    if u.startswith("usd") or u in ("usd", "$"):
        return "USD"
    if u.startswith("eur"):
        return "EUR"
    return None


def normalize_fact(fact: Fact) -> NormalizedFact | None:
    """Produce a NormalizedFact for a validated Fact. Returns None if the fact's value
    isn't numeric (non-numeric facts, e.g. qualitative status changes, aren't normalized
    here and are still retrievable via embeddings on the raw fact)."""
    numeric_value = parse_numeric_value(fact.value)
    if numeric_value is None:
        logger.info("skipping numeric normalization for non-numeric fact id=%s value=%r", fact.id, fact.value)
        return None

    unit = canonical_unit(fact.unit)
    return NormalizedFact(
        fact_id=fact.id,
        entity=clean_entity(fact.entity) or fact.entity.strip(),
        metric=canonical_metric(fact.metric),
        value=numeric_value,
        unit=unit,
        period=canonical_period(fact.period),
        scope=(fact.scope or "unspecified").strip().lower(),
        geography=canonical_geography(getattr(fact, "geography", None)),
        currency=infer_currency(fact.unit),
        reporting_basis=getattr(fact, "reporting_basis", None) or "actual",
        original_value=fact.value,
        original_unit=fact.unit,
    )
