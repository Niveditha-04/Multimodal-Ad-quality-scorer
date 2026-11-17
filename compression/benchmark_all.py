"""Combines the baseline, pruning and quantization results into one benchmark
table (final_benchmark.json) and two plots (results/plots/).

The x-axis of both plots is gzip-compressed size reduction relative to the
baseline checkpoint, not raw size. Unstructured pruning keeps the dense
[128, 1024] and [4, 128] tensor shapes at every sparsity level, so its raw
on-disk size is the same (529,181 bytes) from 30% to 90% sparsity, and a
raw-size axis would put all four points at 0%. Gzip shows the compression
achievable once the zeros are compressed and gives one consistent axis for
every method (see state_dict_gzip_size_bytes in compression/common.py).

Self-implemented and torch.nn.utils.prune pruning are plotted as one
"unstructured pruning" series, since they produce identical masks and
identical accuracy at every sparsity level.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

from compression.common import RESULTS_DIR

PLOTS_DIR = RESULTS_DIR / "plots"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#ececea"


def gzip_compression_pct(gzip_bytes: int, baseline_gzip_bytes: int) -> float:
    return 100.0 * (1 - gzip_bytes / baseline_gzip_bytes)


def load_all_rows():
    baseline = json.loads((RESULTS_DIR / "baseline_metrics.json").read_text())
    self_d = json.loads((RESULTS_DIR / "self_pruning_summary.json").read_text())
    torch_d = json.loads((RESULTS_DIR / "torch_prune_summary.json").read_text())
    structured_d = json.loads((RESULTS_DIR / "structured_pruning_summary.json").read_text())
    quant_d = json.loads((RESULTS_DIR / "quantization_summary.json").read_text())

    baseline_gzip = baseline["gzip_size_bytes"]
    rows = []

    def add(row, group, label):
        rows.append({
            "group": group, "label": label,
            "accuracy": row["accuracy"], "macro_f1": row["macro_f1"],
            "total_parameters": row["total_parameters"], "nonzero_parameters": row["nonzero_parameters"],
            "raw_size_bytes": row["raw_size_bytes"], "gzip_size_bytes": row["gzip_size_bytes"],
            "latency_mean_ms": row["latency_mean_ms"],
            "raw_compression_pct": round(100 * (1 - row["raw_size_bytes"] / baseline["raw_size_bytes"]), 2),
            "gzip_compression_pct": round(gzip_compression_pct(row["gzip_size_bytes"], baseline_gzip), 2),
        })

    add(baseline, "baseline", "baseline (fp32, unpruned)")

    for level in self_d["levels"]:
        add(level["row"], "unstructured", f"unstructured {int(level['target_sparsity'] * 100)}% sparsity")

    for level in structured_d["levels"]:
        add(level["row"], "structured", f"structured {int(level['target_neuron_prune_fraction'] * 100)}% neurons removed")

    add(quant_d["quantized_row"], "quantized",
        f"structured 75% + INT8 quant ({quant_d['quantization_engine']})")

    # torch.prune rows are kept in the table output but not plotted: they are
    # identical to the unstructured rows already added
    torch_rows = [level["row"] for level in torch_d["levels"]]

    return rows, torch_rows, baseline


def print_table(rows):
    header = f"{'label':45} | {'acc':>7} | {'f1':>7} | {'params':>8} | {'raw B':>9} | {'gzip B':>9} | {'lat ms':>8} | {'gzip comp%':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['label']:45} | {r['accuracy']:>7.4f} | {r['macro_f1']:>7.4f} | {r['total_parameters']:>8,} | "
              f"{r['raw_size_bytes']:>9,} | {r['gzip_size_bytes']:>9,} | {r['latency_mean_ms']:>8.4f} | "
              f"{r['gzip_compression_pct']:>9.2f}%")


def make_accuracy_f1_plot(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    baseline = next(r for r in rows if r["group"] == "baseline")

    group_style = {
        "unstructured": (BLUE, "o", "Unstructured pruning (self-implemented = torch.prune)"),
        "structured": (AQUA, "o", "Structured (neuron-level) pruning"),
        "quantized": (ORANGE, "D", "Structured 75% + INT8 quantized"),
    }

    for ax, metric, title in [(axes[0], "accuracy", "Accuracy"), (axes[1], "macro_f1", "Macro F1")]:
        ax.axhline(baseline[metric], color=TEXT_SECONDARY, linestyle="--", linewidth=1, zorder=1,
                   label="baseline (fp32, unpruned)")
        for group, (color, marker, gt_label) in group_style.items():
            pts = [r for r in rows if r["group"] == group]
            xs = [r["gzip_compression_pct"] for r in pts]
            ys = [r[metric] for r in pts]
            ax.scatter(xs, ys, color=color, marker=marker, s=60, zorder=3,
                       edgecolor="white", linewidth=0.6, label=gt_label)
        ax.set_xlabel("gzip-compressed size reduction vs. baseline (%)", color=TEXT_PRIMARY, fontsize=10)
        ax.set_ylabel(title, color=TEXT_PRIMARY, fontsize=10)
        ax.set_title(f"{title} vs. compression", color=TEXT_PRIMARY, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#d9d8d3")
        ax.spines["bottom"].set_color("#d9d8d3")
        ax.tick_params(colors=TEXT_SECONDARY)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.12),
               labelcolor=TEXT_PRIMARY, fontsize=9)
    fig.suptitle("Classifier head: accuracy / F1 vs. compression", color=TEXT_PRIMARY, fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / "accuracy_f1_vs_compression.png", dpi=150, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"saved {PLOTS_DIR / 'accuracy_f1_vs_compression.png'}")


def make_latency_plot(rows):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    baseline = next(r for r in rows if r["group"] == "baseline")

    group_style = {
        "unstructured": (BLUE, "o", "Unstructured pruning (self-implemented = torch.prune)"),
        "structured": (AQUA, "o", "Structured (neuron-level) pruning"),
        "quantized": (ORANGE, "D", "Structured 75% + INT8 quantized"),
    }

    ax.axhline(baseline["latency_mean_ms"], color=TEXT_SECONDARY, linestyle="--", linewidth=1, zorder=1,
               label="baseline (fp32, unpruned)")
    for group, (color, marker, gt_label) in group_style.items():
        pts = [r for r in rows if r["group"] == group]
        xs = [r["gzip_compression_pct"] for r in pts]
        ys = [r["latency_mean_ms"] for r in pts]
        ax.scatter(xs, ys, color=color, marker=marker, s=60, zorder=3,
                   edgecolor="white", linewidth=0.6, label=gt_label)

    ax.set_xlabel("gzip-compressed size reduction vs. baseline (%)", color=TEXT_PRIMARY, fontsize=10)
    ax.set_ylabel("Mean single-sample CPU latency (ms)", color=TEXT_PRIMARY, fontsize=10)
    ax.set_title("Latency vs. compression\n(quantization shrinks size but is slower here -- see README)",
                 color=TEXT_PRIMARY, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d9d8d3")
    ax.spines["bottom"].set_color("#d9d8d3")
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", labelcolor=TEXT_PRIMARY, fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "latency_vs_compression.png", dpi=150, facecolor="white")
    plt.close()
    print(f"saved {PLOTS_DIR / 'latency_vs_compression.png'}")


def main():
    rows, torch_rows, baseline = load_all_rows()
    print_table(rows)

    from compression.common import save_json
    save_json(
        {"rows": rows, "torch_prune_rows_identical_to_unstructured": torch_rows,
         "note": "torch_prune rows are numerically identical to the unstructured rows and are excluded from the plots to avoid overplotting"},
        RESULTS_DIR / "final_benchmark.json",
    )

    make_accuracy_f1_plot(rows)
    make_latency_plot(rows)


if __name__ == "__main__":
    main()
