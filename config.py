"""
config.py

Loads environment variables and defines all project-wide constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()  # Load variables from .env


# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------

GROQ_API_KEY        = os.environ["GROQ_API_KEY"]
REDIS_URL           = os.environ["REDIS_URL"]
LANGFUSE_PUBLIC_KEY = os.environ["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_SECRET_KEY = os.environ["LANGFUSE_SECRET_KEY"]
LANGFUSE_HOST       = os.environ.get(
    "LANGFUSE_HOST",
    "https://cloud.langfuse.com"
)


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"


# ---------------------------------------------------------------------------
# Raw source files
# ---------------------------------------------------------------------------

CSV_PATH      = DATA_DIR / "startup_funding_clean.csv"
PROFILES_PATH = DATA_DIR / "company_profiles.json"


# ---------------------------------------------------------------------------
# Generated artifacts
# ---------------------------------------------------------------------------

CHROMA_PATH = PROCESSED_DIR / "funding_db"


# ---------------------------------------------------------------------------
# Runtime files
# ---------------------------------------------------------------------------

LOG_FILE        = BASE_DIR / "chatbot.log"
SESSION_ID_FILE = BASE_DIR / ".session_id"


# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------

GROQ_MODEL        = "llama-3.3-70b-versatile"
EMBED_MODEL       = "all-MiniLM-L6-v2"
CHROMA_COLLECTION = "startups"


# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------

QUALITY_THRESHOLD  = 0.35
FALLBACK_THRESHOLD = 0.20
CHROMA_TOP_K       = 5
HISTORY_WINDOW     = 6

MAX_MENTIONED_COMPANIES = 10


# ---------------------------------------------------------------------------
# Default fallback message
# ---------------------------------------------------------------------------

FALLBACK_RESPONSE = (
    "I couldn’t find relevant information"
    "Try mentioning a specific startup name, sector (fintech, edtech, logistics), "
    "or city (Bangalore, Mumbai, Delhi) to refine the search."
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_files():
    """
    Ensure required runtime files exist.
    Called by store modules to fail fast during startup.
    """
    missing = []

    if not CSV_PATH.exists():
        missing.append(
            f"{CSV_PATH}  → make sure startup_funding_clean.csv is in data/"
        )

    if not CHROMA_PATH.exists():
        missing.append(
            f"{CHROMA_PATH}  → run: python -m db.build_chroma"
        )

    if missing:
        raise FileNotFoundError(
            "Required files missing:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )