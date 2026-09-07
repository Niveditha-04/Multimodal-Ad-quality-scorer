"""Step 1 of the compression study: measure the CURRENT trained classifier
head (model/classifier_head.pt, already trained by model/train_classifier.py
--modality joint) before any pruning or quantization touches it. Every later
compression result in this study is measured against these exact numbers.

Cross-checks its own accuracy/F1 against model/results/metrics.json (the
number already reported in the top-level README) as a sanity check that
this script's evaluation logic is loading the same split, same model, and
producing the same result -- not a subtly different computation.
"""
import json
from pathlib import Path

from compression.common import RESULTS_DIR, full_benchmark_row, load_baseline_head, load_split_data, save_json


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_split_data()
    print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    model = load_baseline_head()
    row = full_benchmark_row(model, X_test, y_test, name="baseline_fp32")

    print(f"\nbaseline accuracy: {row['accuracy']:.4f}")
    print(f"baseline macro F1: {row['macro_f1']:.4f}")
    print(f"total parameters:  {row['total_parameters']:,}")
    print(f"raw state_dict size:  {row['raw_size_bytes']:,} bytes")
    print(f"gzip state_dict size: {row['gzip_size_bytes']:,} bytes")
    print(f"latency (batch=1, CPU): mean={row['latency_mean_ms']:.4f}ms "
          f"median={row['latency_median_ms']:.4f}ms p95={row['latency_p95_ms']:.4f}ms")

    existing = json.loads(Path("model/results/metrics.json").read_text())
    print(f"\ncross-check against model/results/metrics.json:")
    print(f"  metrics.json accuracy = {existing['accuracy']:.6f}")
    print(f"  this script accuracy  = {row['accuracy']:.6f}")
    match = abs(existing["accuracy"] - row["accuracy"]) < 1e-9
    print(f"  exact match: {match}")
    if not match:
        raise RuntimeError(
            "Baseline accuracy computed here does not match model/results/metrics.json -- "
            "stop and investigate before trusting any downstream compression comparison."
        )

    save_json(row, path=RESULTS_DIR / "baseline_metrics.json")
    save_json({"cross_check_metrics_json_accuracy": existing["accuracy"],
               "cross_check_this_script_accuracy": row["accuracy"],
               "exact_match": match},
              path=RESULTS_DIR / "baseline_cross_check.json")


if __name__ == "__main__":
    main()
