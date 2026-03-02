"""
tests/test_llama.py

Manual tests for LlamaIndex structured query engine.

Run:
    python -m tests.test_llama
"""

from stores.llama_store import query_llama, is_error_response


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run_test(query: str, check_fn=None, label: str = "") -> bool:
    """Run a structured query and optionally validate the result."""
    ans = query_llama(query)

    if is_error_response(ans):
        print(f"FAIL  {label or query}")
        print(f"  error: {ans[:150]}\n")
        return False

    passed = check_fn(ans) if check_fn else True
    status = "PASS" if passed else "CHECK"

    print(f"{status}  {label or query}")
    print(f"  answer: {ans[:200]}\n")

    return passed


# ---------------------------------------------------------------------------
# Test Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("LlamaIndex Structured Query Tests\n")

    results = []

    # Investor queries
    results.append(run_test(
        "Who invested in Razorpay?",
        check_fn=lambda a: any(w in a.lower()
                               for w in ["sequoia", "tiger", "ribbit", "y combinator", "gic"]),
        label="Razorpay investors",
    ))

    results.append(run_test(
        "Who invested in Swiggy?",
        label="Swiggy investors",
    ))

    # Funding round queries
    results.append(run_test(
        "What funding rounds did Swiggy raise?",
        check_fn=lambda a: any(w in a.lower()
                               for w in ["series", "round", "seed"]),
        label="Swiggy rounds",
    ))

    results.append(run_test(
        "How much did Razorpay raise in each round?",
        check_fn=lambda a: "$" in a
                           or "usd" in a.lower()
                           or "million" in a.lower(),
        label="Razorpay round amounts",
    ))

    # Aggregation queries
    results.append(run_test(
        "How many fintech startups are in Bangalore?",
        check_fn=lambda a: any(c.isdigit() for c in a),
        label="Count fintech in Bangalore",
    ))

    results.append(run_test(
        "Which sector has the most startups?",
        label="Sector with most startups",
    ))

    results.append(run_test(
        "Which city has the most startups?",
        label="City with most startups",
    ))

    results.append(run_test(
        "How many seed rounds happened in 2018?",
        check_fn=lambda a: any(c.isdigit() for c in a),
        label="Seed rounds 2018",
    ))

    # Investor-based filtering
    results.append(run_test(
        "List all startups funded by Sequoia",
        check_fn=lambda a: len(a) > 20,
        label="Startups funded by Sequoia",
    ))

    passed = sum(results)
    total = len(results)

    print(f"Results: {passed}/{total} {'PASS' if passed == total else 'CHECK'}")