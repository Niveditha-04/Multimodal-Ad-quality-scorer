"""Runs the adversarial cases through the real rag.explain.generate_explanation()
pipeline and saves the raw output. guardrails/results_before_defense.json is a
run before the prompt hardening and output validator were added, and
guardrails/results_after_defense.json is a run after.
"""
import json
import sys

from PIL import Image

from guardrails.adversarial_cases import ADVERSARIAL_CASES
from rag.explain import generate_explanation

TEST_IMAGE = "data/images/0000_samoyed_64.jpg"


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "guardrails/results.json"
    image = Image.open(TEST_IMAGE).convert("RGB")

    results = []
    for case in ADVERSARIAL_CASES:
        explanation = generate_explanation(
            ad_text=case["ad_text"],
            predicted_label=case["predicted_label"],
            image=image,
        )
        results.append({
            "name": case["name"],
            "ad_text": case["ad_text"],
            "predicted_label": case["predicted_label"],
            "explanation": explanation,
        })
        print(f"=== {case['name']} ===")
        print(explanation)
        print()

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
