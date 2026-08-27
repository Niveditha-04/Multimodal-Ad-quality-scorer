"""Wilson score 95% confidence intervals for the ablation accuracy/recall
numbers. With 24 examples per class, a point estimate like 0.292 recall
carries real sampling uncertainty -- this makes that uncertainty explicit
instead of letting a decimal like "29.2%" read as more precise than a
24-sample count actually supports.
"""
import json
import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def main():
    results = {}
    for modality in ["image", "text", "joint"]:
        r = json.loads(open(f"model/results/{modality}_results.json").read())
        n_test = r["n_test"]
        n_correct = round(r["accuracy"] * n_test)
        acc_lo, acc_hi = wilson_interval(n_correct, n_test)

        per_class_ci = {}
        for label, stats in r["per_class"].items():
            support = stats["support"]
            n_recalled = round(stats["recall"] * support)
            lo, hi = wilson_interval(n_recalled, support)
            per_class_ci[label] = {
                "recall": stats["recall"],
                "recall_95ci": [round(lo, 3), round(hi, 3)],
                "support": support,
            }

        results[modality] = {
            "accuracy": r["accuracy"],
            "accuracy_95ci": [round(acc_lo, 3), round(acc_hi, 3)],
            "n_test": n_test,
            "per_class_recall": per_class_ci,
        }

        print(f"=== {modality} ===")
        print(f"  accuracy: {r['accuracy']:.3f}  95% CI: [{acc_lo:.3f}, {acc_hi:.3f}]  (n={n_test})")
        for label, ci in per_class_ci.items():
            print(f"  {label:18s} recall={ci['recall']:.3f}  95% CI: [{ci['recall_95ci'][0]:.3f}, {ci['recall_95ci'][1]:.3f}]  (n={ci['support']})")
        print()

    with open("model/results/confidence_intervals.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved to model/results/confidence_intervals.json")


if __name__ == "__main__":
    main()
