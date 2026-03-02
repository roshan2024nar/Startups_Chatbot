"""
utils/data_loader.py

Centralized data loading for raw source files.
All modules should load data through these functions.

Loaded files:
- startup_funding_clean.csv  → pandas DataFrame
- company_profiles.json      → list of dicts

"""

import json
import pandas as pd
from pathlib import Path

from config import CSV_PATH, PROFILES_PATH


# Required CSV columns
REQUIRED_CSV_COLUMNS = {
    "sr_no", "date", "startup_name", "subvertical",
    "city", "investors", "investment_type", "amount_usd",
    "remarks", "year", "industry",
}

# Required profile keys for ChromaDB
REQUIRED_PROFILE_KEYS = {"startup_name", "embedding_text", "industry"}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_csv(path: Path = None, validate: bool = True) -> pd.DataFrame:
    """Load funding CSV with optional schema validation."""

    p = Path(path) if path else CSV_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"CSV not found: {p}\n"
            "Ensure startup_funding_clean.csv exists in data/"
        )

    df = pd.read_csv(p)

    if validate:
        missing = REQUIRED_CSV_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}\n"
                f"Found columns: {df.columns.tolist()}")

    return df


def load_profiles(path: Path = None, validate: bool = True) -> list[dict]:
    """Load company_profiles.json with optional key validation."""

    p = Path(path) if path else PROFILES_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"company_profiles.json not found: {p}\n")

    with open(p, encoding="utf-8") as f:
        profiles = json.load(f)

    if not isinstance(profiles, list):
        raise ValueError(
            f"company_profiles.json must be a JSON array, got {type(profiles)}"
        )

    if validate:
        invalid = []
        for i, profile in enumerate(profiles):
            missing = REQUIRED_PROFILE_KEYS - set(profile.keys())
            if missing:
                invalid.append((i, profile.get("startup_name", "?"), missing))

        if invalid:
            sample = invalid[:3]
            raise ValueError(
                f"{len(invalid)} profiles missing required keys:\n"
                + "\n".join(f"[{i}] {name}: {keys}" for i, name, keys in sample)
                + ("\n..." if len(invalid) > 3 else "")
            )

    return profiles


def load_all(validate: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """Load both CSV and profiles in one call."""

    return load_csv(validate=validate), load_profiles(validate=validate)


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def describe_csv(df: pd.DataFrame) -> None:
    """Print a quick summary of the funding CSV."""
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print(f"Industries: {sorted(df['industry'].dropna().unique().tolist())}")
    print(f"Cities: {sorted(df['city'].dropna().unique().tolist())[:10]} ...")
    print(f"Year range: {int(df['year'].min())} – {int(df['year'].max())}")
    print(f"Null counts: {df.isnull().sum().to_dict()}")


def describe_profiles(profiles: list[dict]) -> None:
    """Print a quick summary of company profiles."""
    missing_text = [p["startup_name"] for p in profiles if not p.get("embedding_text")]
    missing_ind  = [p["startup_name"] for p in profiles if not p.get("industry")]

    print(f"Profiles: {len(profiles)}")
    print(f"Missing embedding_text: {len(missing_text)}")
    print(f"Missing industry: {len(missing_ind)}")

    if missing_text:
        print(f"Fix examples: {missing_text[:10]}")