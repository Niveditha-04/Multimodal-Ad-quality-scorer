"""Combines the baseline, pruning and quantization results into one benchmark
table (final_benchmark.json). Unstructured pruning keeps the dense [128, 1024]
and [4, 128] tensor shapes at every sparsity level, so its raw file size equals
the baseline's, and the gzip columns show the compression achievable once the
zeros are compressed (see state_dict_gzip_size_bytes in compression/common.py).

Self-implemented and torch.nn.utils.prune pruning give identical masks and
identical accuracy, so the table lists the self-implemented rows and keeps the
torch.nn.utils.prune rows separately in the JSON.
"""
import json
from pathlib import Path

from compression.common import RESULTS_DIR


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

    qrow = quant_d["quantized_row"]
    src = f"{qrow['source_method']} {qrow['source_level_pct']}%"
    add(qrow, "quantized", f"{src} + INT8 quant ({quant_d['quantization_engine']})")

    # torch.prune rows are kept in the JSON only: they are identical to the
    # unstructured rows already added
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



def main():
    rows, torch_rows, baseline = load_all_rows()
    print_table(rows)

    from compression.common import save_json
    save_json(
        {"rows": rows, "torch_prune_rows_identical_to_unstructured": torch_rows,
         "note": "torch_prune rows are numerically identical to the unstructured rows and are listed separately"},
        RESULTS_DIR / "final_benchmark.json",
    )



if __name__ == "__main__":
    main()
