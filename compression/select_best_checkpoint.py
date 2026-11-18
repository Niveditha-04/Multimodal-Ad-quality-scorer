"""Picks the best pruned and fine-tuned checkpoint across the three pruning
experiments, to pass to compression/quantize_int8.py. All 12 candidates are
ranked by accuracy on the validation split (descending), ties are broken by
gzip state_dict size (ascending), and the top row wins. Test accuracy is
reported next to each candidate but never used for ranking. Every candidate is
written to best_checkpoint_selection.json so the choice can be audited.
"""
import json
from pathlib import Path

from compression.common import RESULTS_DIR, save_json

SUMMARY_FILES = {
    "self_implemented": (RESULTS_DIR / "self_pruning_summary.json", "self_pruned_{pct}pct.pt"),
    "torch_prune": (RESULTS_DIR / "torch_prune_summary.json", "torch_prune_{pct}pct.pt"),
    "structured": (RESULTS_DIR / "structured_pruning_summary.json", "structured_pruned_{pct}pct.pt"),
}

LEVEL_KEY = {
    "self_implemented": "target_sparsity",
    "torch_prune": "target_sparsity",
    "structured": "target_neuron_prune_fraction",
}


def load_candidates():
    candidates = []
    for method, (summary_path, ckpt_template) in SUMMARY_FILES.items():
        data = json.loads(summary_path.read_text())
        for level in data["levels"]:
            row = level["row"]
            pct = int(round(level[LEVEL_KEY[method]] * 100))
            candidates.append({
                "method": method,
                "level_pct": pct,
                "checkpoint_path": f"compression/checkpoints/{ckpt_template.format(pct=pct)}",
                "val_accuracy": row["val_accuracy"],
                "n_val": row["n_val"],
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
                "total_parameters": row["total_parameters"],
                "nonzero_parameters": row["nonzero_parameters"],
                "raw_size_bytes": row["raw_size_bytes"],
                "gzip_size_bytes": row["gzip_size_bytes"],
                "latency_mean_ms": row["latency_mean_ms"],
                "is_structured": method == "structured",
            })
    return candidates


def main():
    candidates = load_candidates()
    candidates.sort(key=lambda c: (-c["val_accuracy"], c["gzip_size_bytes"]))

    print(f"{'method':>16} | {'level':>6} | {'val_acc':>7} | {'test_acc':>8} | {'gzip_bytes':>10} | {'params':>8}")
    for c in candidates:
        print(f"{c['method']:>16} | {c['level_pct']:>5}% | {c['val_accuracy']:>7.4f} | {c['accuracy']:>8.4f} | "
              f"{c['gzip_size_bytes']:>10,} | {c['total_parameters']:>8,}")

    best = candidates[0]
    print(f"\nselected: {best['method']} @ {best['level_pct']}% -> {best['checkpoint_path']}")
    print(f"reason: highest validation accuracy ({best['val_accuracy']:.4f}, n={best['n_val']}) among all 12 "
          f"pruned+fine-tuned checkpoints, tie-broken by smaller gzip size. "
          f"Its test accuracy is {best['accuracy']:.4f}.")

    save_json({"all_candidates_ranked": candidates, "selected": best}, RESULTS_DIR / "best_checkpoint_selection.json")
    return best


if __name__ == "__main__":
    main()
