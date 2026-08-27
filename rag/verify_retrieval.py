"""Sanity-checks retrieve_context() against three realistic flagged-ad
queries, one per non-approved label. Run after any change to
build_vector_store.py or retrieve.py's ranking logic.
"""
from rag.retrieve import retrieve_context

TEST_CASES = [
    ("This dog food CURES joint pain in dogs within days -- guaranteed or your money back. Only 2 left in stock!", "policy_violation"),
    ("Our interactive feather wand toy is formulated with cat nutrition in mind -- a solid pick for Persians at any age.", "misleading"),
    ("PRODUCT SALE SALE SALE!!! best!! dog cheap price fast shipping click", "low_quality"),
]


def main() -> None:
    for text, label in TEST_CASES:
        print("=" * 80)
        print(f"QUERY (predicted={label}): {text}")
        ctx = retrieve_context(text, predicted_label=label)
        print("policy rules:")
        for r in ctx["policy_rules"]:
            d = f"{r['distance']:.3f}" if r["distance"] is not None else "GUARANTEED"
            print(f"  [{d}] {r['category_name']}")
        print("similar examples:")
        for e in ctx["examples"]:
            print(f"  [{e['distance']:.3f}] ({e['ground_truth_label']}) {e['ad_text'][:90]}")
        print()


if __name__ == "__main__":
    main()
