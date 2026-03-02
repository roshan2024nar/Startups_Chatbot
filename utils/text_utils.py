"""
utils/text_utils.py

normalization helpers, and text utilities.

"""
# ---------------------------------------------------------------------------
# Industry keyword
# ---------------------------------------------------------------------------

INDUSTRY_KW: dict[str, str] = {
    # Fintech
    "fintech": "fintech",
    "payment": "fintech",
    "insurance": "fintech",
    "lending": "fintech",
    "nbfc": "fintech",
    "wealth": "fintech",

    # Edtech
    "edtech": "edtech",
    "education": "edtech",
    "learning": "edtech",
    "coaching": "edtech",

    # Food
    "food": "food",
    "grocery": "food",
    "restaurant": "food",
    "beverage": "food",

    # Logistics
    "logistics": "logistics",
    "freight": "logistics",
    "supply chain": "logistics",
    "warehou": "logistics",

    # Healthcare
    "health": "healthcare",
    "healthcare": "healthcare",
    "medical": "healthcare",
    "pharma": "healthcare",
    "diagnostic": "healthcare",

    # Ecommerce
    "ecommerce": "ecommerce",
    "retail": "ecommerce",
    "marketplace": "ecommerce",

    # SaaS
    "saas": "saas",

    # Real estate
    "real estate": "real_estate",
    "property": "real_estate",

    # Mobility
    "mobility": "mobility",
    "transport": "mobility",
    "cab": "mobility",
    "ride": "mobility",

    # Media / entertainment
    "media": "media_entertainment",
    "gaming": "media_entertainment",
    "entertainment": "media_entertainment",

    # Technology (generic)
    "technology": "technology",
}


# ---------------------------------------------------------------------------
# City keyword → canonical city label
# ---------------------------------------------------------------------------

CITY_KW: dict[str, str] = {
    "bangalore": "bangalore",
    "bengaluru": "bangalore",
    "banglore":  "bangalore",

    "mumbai": "mumbai",
    "bombay": "mumbai",

    "delhi":     "delhi",
    "new delhi": "delhi",

    "gurgaon":  "gurgaon",
    "gurugram": "gurgaon",

    "hyderabad": "hyderabad",
    "pune":      "pune",
    "chennai":   "chennai",
    "noida":     "noida",
    "kolkata":   "kolkata",
}

# ---------------------------------------------------------------------------
# Safe type helpers
# ---------------------------------------------------------------------------

def safe_str(val) -> str:
    """Return string value or empty string if None."""
    return "" if val is None else str(val)


def safe_float(val) -> float:
    """Return float value or 0.0 if conversion fails."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Context builder for LLM
# ---------------------------------------------------------------------------

def build_context(results: list[dict]) -> str:
    """
    Combine Chroma search results into a single context string
    passed to the LLM.
    """
    return "\n\n".join(
        f"[score={r['score']}] {r['embedding_text']}"
        for r in results
    )