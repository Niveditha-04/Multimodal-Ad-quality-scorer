"""Picks the single best pruned + fine-tuned checkpoint across all three
pruning experiments (Steps 2-4), to hand to Step 6 (INT8 quantization).
Selection is programmatic, not eyeballed: rank all 12 pruned+fine-tuned
checkpoints by post-fine-tune accuracy (descending), tie-break by gzip
state_dict size (ascending -- smaller is better among equally-accurate
candidates), and take the top row. Every candidate and the reasoning is
written to disk so the pick is auditable, not asserted.
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
    candidates.sort(key=lambda c: (-c["accuracy"], c["gzip_size_bytes"]))

    print(f"{'method':>16} | {'level':>6} | {'accuracy':>8} | {'macro_f1':>8} | {'gzip_bytes':>10} | {'params':>8}")
    for c in candidates:
        print(f"{c['method']:>16} | {c['level_pct']:>5}% | {c['accuracy']:>8.4f} | {c['macro_f1']:>8.4f} | "
              f"{c['gzip_size_bytes']:>10,} | {c['total_parameters']:>8,}")

    best = candidates[0]
    print(f"\nselected: {best['method']} @ {best['level_pct']}% -> {best['checkpoint_path']}")
    print(f"reason: highest post-fine-tune accuracy ({best['accuracy']:.4f}) among all 12 pruned+fine-tuned "
          f"checkpoints, tie-broken by smaller gzip size")

    save_json({"all_candidates_ranked": candidates, "selected": best}, RESULTS_DIR / "best_checkpoint_selection.json")
    return best


if __name__ == "__main__":
    main()
