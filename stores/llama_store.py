"""
stores/llama_store.py

Structured query layer using LlamaIndex PandasQueryEngine
over startup_funding_clean.csv.

Used for aggregation, counts, investor lookups, and other.
"""

import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.groq import Groq as LlamaGroq

from config import GROQ_API_KEY, GROQ_MODEL, validate_files
from utils.data_loader import load_csv


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

validate_files()  # Ensure required files exist

df = load_csv()  # Load funding CSV into memory

llm = LlamaGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY) 

# PandasQueryEngine generates pandas code and executes it on the DataFrame

engine = PandasQueryEngine(
    df=df,
    llm=llm,
    verbose=False,            # Enable for debugging generated pandas code
    synthesize_response=True, # Summarize raw result into natural language
)


def query_llama(question: str) -> str:
    """
    Execute a structured query against the CSV using LlamaIndex.
    Returns natural language output or "__error__:<message>" on failure.
    """
    try:
        response = engine.query(question) # Generate pandas code via LLM and execute
        return str(response).strip()
    
    except Exception as e:
        return f"__error__:{e}"


def is_error_response(ans: str) -> bool:
    """
    Detect whether response should be treated as a failure.
    """
    if not ans or len(ans) < 5:
        return True

    if ans.startswith("__error__"):
        return True

    # Basic low-quality detection
    if any(w in ans.lower() for w in ["i don't know", "cannot", "no information"]):
        return True

    return False