"""
tests/test_chroma.py

Basic manual tests for ChromaDB semantic and filtered search.

Run:
    python -m tests.test_chroma
"""

from stores.chroma_store import (
    semantic_search,
    filtered_search,
    get_collection_count,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_semantic_test(query: str, expected_industry: str,
                    top_k: int = 5, threshold: float = 0.3) -> bool:
    
    """Check that most semantic results match the expected industry."""
    results = semantic_search(query, top_k=top_k, threshold=threshold)

    industry_match = sum(
        1 for r in results
        if r["metadata"].get("industry") == expected_industry
    )

    passed = industry_match >= 2
    print(f"{'PASS' if passed else 'FAIL'}  semantic: '{query}'")
    print(f"  expected industry: {expected_industry}")
    print(f"  match count: {industry_match}/{len(results)}\n")

    return passed


def run_gibberish_test(query: str, threshold: float = 0.5) -> bool:
    """Ensure nonsense query returns no results."""
    results = semantic_search(query, top_k=5, threshold=threshold)
    passed = len(results) == 0

    print(f"{'PASS' if passed else 'FAIL'}  gibberish: '{query}' → {len(results)} results\n")
    return passed


def run_filtered_test(query: str, filters: dict, top_k: int = 5) -> bool:
    """Ensure filtered search respects metadata constraints."""
    results = filtered_search(query, filters=filters, top_k=top_k)

    if not results:
        print(f"FAIL  filtered: '{query}'  filters={filters} (no results)\n")
        return False

    industry_ok = all(
        r["metadata"].get("industry") == filters["industry"]
        for r in results
    ) if filters.get("industry") else True

    city_ok = all(
        r["metadata"].get("city") == filters["city"]
        for r in results
    ) if filters.get("city") else True

    passed = industry_ok and city_ok

    print(f"{'PASS' if passed else 'FAIL'}  filtered: '{query}'  filters={filters}\n")
    return passed


# ---------------------------------------------------------------------------
# Test Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("ChromaDB Search Tests")
    print(f"Collection size: {get_collection_count()} companies\n") # Verify if the data is in collection

    results = []

    # Semantic tests
    results += [
        run_semantic_test("payments startup", "fintech"),
        run_semantic_test("food delivery company", "food"),
        run_semantic_test("edtech learning platform", "edtech"),
        run_semantic_test("logistics startup", "logistics"),
        run_gibberish_test("xyzqwerty asdfghjkl"),
    ]

    # Filtered tests
    results += [
        run_filtered_test("payments company", {"industry": "fintech"}),
        run_filtered_test("food delivery", {"industry": "food"}),
        run_filtered_test("startup", {"city": "bangalore"}),
        run_filtered_test("delivery company", {"industry": "food", "city": "bangalore"}),
        run_filtered_test("payments platform", {"industry": "fintech", "city": "bangalore"}),
        run_filtered_test("learning platform", {"industry": "edtech", "city": "bangalore"}),
    ]

    passed = sum(results)
    total = len(results)

    print(f"Results: {passed}/{total} {'PASS' if passed == total else 'FAIL'}")