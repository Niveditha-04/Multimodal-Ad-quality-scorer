"""Computes aggregate scores from eval/scores.py against eval/golden_set.json.
Cross-checks that every golden-set ad has a score (no silent omissions) and
computes the numbers programmatically rather than by hand, to catch any
arithmetic slip in the manual scoring pass.
"""
import json
import math

from eval.scores import SCORES

AXES = ["groundedness", "rule_accuracy", "hallucination"]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def main() -> None:
    golden_set = json.loads(open("eval/golden_set.json").read())
    golden_ids = {e["ad_id"] for e in golden_set}
    scored_ids = set(SCORES.keys())

    missing = golden_ids - scored_ids
    extra = scored_ids - golden_ids
    assert not missing, f"golden set ads with no score: {missing}"
    assert not extra, f"scores for ads not in golden set: {extra}"

    per_example = []
    for e in golden_set:
        g, r, h, note = SCORES[e["ad_id"]]
        per_example.append({
            "ad_id": e["ad_id"], "ground_truth_label": e["ground_truth_label"],
            "model_predicted_label": e["model_predicted_label"],
            "is_misclassification": e["is_misclassification"],
            "groundedness": g, "rule_accuracy": r, "hallucination": h, "note": note,
        })

    n = len(per_example)
    aggregate = {}
    for axis in AXES:
        values = [ex[axis] for ex in per_example]
        aggregate[axis] = {
            "mean": round(sum(values) / n, 3),
            "n_score_0": sum(1 for v in values if v == 0),
            "n_score_1": sum(1 for v in values if v == 1),
            "n_score_2": sum(1 for v in values if v == 2),
            "failure_rate_pct": round(100 * sum(1 for v in values if v == 0) / n, 1),
        }

    misclass = [ex for ex in per_example if ex["is_misclassification"]]
    true_pos = [ex for ex in per_example if not ex["is_misclassification"]]

    def summarize(subset, label):
        out = {}
        for axis in AXES:
            values = [ex[axis] for ex in subset]
            out[axis] = {
                "mean": round(sum(values) / len(subset), 3) if subset else None,
                "n_score_0": sum(1 for v in values if v == 0),
            }
        print(f"{label} (n={len(subset)}):", out)
        return out

    hallu_lo, hallu_hi = wilson_interval(aggregate["hallucination"]["n_score_0"], n)

    results = {
        "n_examples": n,
        "aggregate": aggregate,
        "headline_hallucination_rate_pct": aggregate["hallucination"]["failure_rate_pct"],
        "headline_hallucination_rate_95ci_pct": [round(hallu_lo * 100, 1), round(hallu_hi * 100, 1)],
        "by_misclassification_status": {
            "misclassified_ads": summarize(misclass, "misclassified"),
            "true_positive_ads": summarize(true_pos, "true_positive"),
        },
        "per_example": per_example,
    }

    print("\n=== aggregate ===")
    for axis in AXES:
        print(f"{axis}: mean={aggregate[axis]['mean']}  0s={aggregate[axis]['n_score_0']} "
              f"1s={aggregate[axis]['n_score_1']} 2s={aggregate[axis]['n_score_2']} "
              f"failure_rate={aggregate[axis]['failure_rate_pct']}%")

    with open("eval/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved to eval/results.json ({n} examples)")


if __name__ == "__main__":
    main()
