"""Picks the pruned and fine-tuned checkpoint to deploy, to pass to
compression/quantize_int8.py. A candidate is eligible if its validation
accuracy is at least the baseline's (baseline_metrics.json). Among the
eligible candidates the smallest raw file size wins (the size that is actually
deployed), ties are broken by higher validation accuracy, then smaller gzip
size. Test accuracy is reported next to each candidate but never used for
selection. Every candidate is written to best_checkpoint_selection.json so the
choice can be audited.
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
    baseline = json.loads((RESULTS_DIR / "baseline_metrics.json").read_text())
    bar = baseline["val_accuracy"]
    candidates = load_candidates()
    for c in candidates:
        c["meets_baseline_val"] = c["val_accuracy"] >= bar

    eligible = [c for c in candidates if c["meets_baseline_val"]]
    if not eligible:
        raise RuntimeError(f"no candidate reaches the baseline's validation accuracy ({bar:.4f})")
    eligible.sort(key=lambda c: (c["raw_size_bytes"], -c["val_accuracy"], c["gzip_size_bytes"], c["method"]))
    rest = sorted((c for c in candidates if not c["meets_baseline_val"]), key=lambda c: -c["val_accuracy"])
    ranked = eligible + rest

    best_val = max(c["val_accuracy"] for c in candidates)
    n_tied = sum(1 for c in candidates if c["val_accuracy"] == best_val)
    print(f"baseline validation accuracy: {bar:.4f} (n={baseline['n_val']})")
    print(f"{'method':>16} | {'level':>6} | {'val_acc':>7} | {'eligible':>8} | {'raw_bytes':>9} | {'test_acc':>8} | {'params':>8}")
    for c in ranked:
        print(f"{c['method']:>16} | {c['level_pct']:>5}% | {c['val_accuracy']:>7.4f} | {str(c['meets_baseline_val']):>8} | "
              f"{c['raw_size_bytes']:>9,} | {c['accuracy']:>8.4f} | {c['total_parameters']:>8,}")

    best = ranked[0]
    print(f"\nselected: {best['method']} @ {best['level_pct']}% -> {best['checkpoint_path']}")
    print(f"reason: smallest raw file ({best['raw_size_bytes']:,} B) among the {len(eligible)} candidates whose validation "
          f"accuracy is at least the baseline's ({bar:.4f}). {n_tied} of {len(candidates)} candidates tie at the best "
          f"validation accuracy ({best_val:.4f}). Its validation accuracy is {best['val_accuracy']:.4f}.")

    save_json(
        {"rule": "smallest raw file among candidates with validation accuracy >= baseline's; "
                 "ties by validation accuracy, then gzip size",
         "baseline_val_accuracy": bar, "n_val": baseline["n_val"],
         "n_eligible": len(eligible), "n_tied_at_best_val_accuracy": n_tied, "best_val_accuracy": best_val,
         "all_candidates_ranked": ranked, "selected": best},
        RESULTS_DIR / "best_checkpoint_selection.json",
    )
    return best


if __name__ == "__main__":
    main()
