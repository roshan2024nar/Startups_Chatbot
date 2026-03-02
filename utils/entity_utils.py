"""
utils/entity_utils.py

Helper functions for extracting company names from text.
"""

import re
import logging

from stores.chroma_store import get_all_company_names


# Cached set of all known company names
ALL_COMPANY_NAMES = None
LOGGER = logging.getLogger("chatbot")


def load_company_names() -> set:
    """Load company names once and cache them."""
    global ALL_COMPANY_NAMES

    if ALL_COMPANY_NAMES is None:
        try:
            names = get_all_company_names()
            ALL_COMPANY_NAMES = set(names)
            LOGGER.info(f"Loaded {len(ALL_COMPANY_NAMES)} company names.")
        except Exception as e:
            LOGGER.error(f"Error loading company names: {e}")
            ALL_COMPANY_NAMES = set()

    return ALL_COMPANY_NAMES


def extract_company_names(text: str) -> list[str]:
    """
    Return list of known company names found in the given text.
    Uses word-boundary matching to avoid partial matches.
    """
    if not text:
        return []

    names = load_company_names()
    if not names:
        return []

    found = []
    text_lower = text.lower()

    for name in names:
        # Escape special regex characters in name (e.g., "Byju's")
        pattern = r"\b" + re.escape(name) + r"\b"

        if re.search(pattern, text_lower, re.IGNORECASE):
            found.append(name)

    return found