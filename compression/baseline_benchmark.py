"""Measures the trained classifier head (model/classifier_head.pt) before any
pruning or quantization: accuracy, macro F1, parameter count, state_dict size
(raw and gzip-compressed) and single-sample CPU latency. Every compression
result is compared against these numbers.

The accuracy is cross-checked against model/results/joint_results.json, so the
script is known to load the same split and model as the original training
run. It raises if the two differ.
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

    existing = json.loads(Path("model/results/joint_results.json").read_text())
    print(f"\ncross-check against model/results/joint_results.json:")
    print(f"  joint_results.json accuracy = {existing['accuracy']:.6f}")
    print(f"  this script accuracy  = {row['accuracy']:.6f}")
    match = abs(existing["accuracy"] - row["accuracy"]) < 1e-9
    print(f"  exact match: {match}")
    if not match:
        raise RuntimeError(
            "Baseline accuracy does not match model/results/joint_results.json. "
            "Check the split and checkpoint before comparing compressed models."
        )

    save_json(row, path=RESULTS_DIR / "baseline_metrics.json")


if __name__ == "__main__":
    main()
