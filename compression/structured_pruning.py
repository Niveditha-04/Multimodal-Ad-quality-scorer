"""Structured (neuron-level) pruning of the classifier head.

Unstructured pruning (manual_magnitude_pruning.py, torch_prune_pruning.py)
zeros individual weights but keeps the dense [128, 1024] and [4, 128] tensor
shapes, so the saved state_dict and the matmul cost do not shrink without
sparse storage or kernels. Structured pruning removes whole hidden units: the
model has smaller net.0.weight and net.3.weight tensors, fewer parameters, a
smaller state_dict and a faster forward pass.

Neuron importance is the L2 norm of the neuron's incoming weight row
(net.0.weight[i, :]) plus the L2 norm of its outgoing weight column
(net.3.weight[:, i]). The lowest-importance neurons are removed and the
survivors' rows and columns are copied into a smaller MLPHead (reduced
hidden_dim), so nothing is masked.

As in the other pruning scripts, each level is pruned one-shot from the
baseline, fine-tuned, and recorded with an accuracy-recovery curve.
"""
import torch

from compression.common import (
    CHECKPOINT_DIR, RESULTS_DIR, build_head, evaluate, fine_tune_with_mask,
    full_benchmark_row, load_split_data, save_json,
)
from model.classifier_architecture import HIDDEN_DIM

NEURON_PRUNE_FRACTIONS = [0.25, 0.5, 0.75, 0.90]


def compute_neuron_importance(state_dict: dict) -> torch.Tensor:
    w_in = state_dict["net.0.weight"]   # [hidden_dim, input_dim] -- row i feeds INTO hidden unit i
    w_out = state_dict["net.3.weight"]  # [n_classes, hidden_dim] -- column i comes FROM hidden unit i
    return w_in.norm(dim=1) + w_out.norm(dim=0)


def build_structurally_pruned_head(baseline_state: dict, target_hidden_dim: int):
    importance = compute_neuron_importance(baseline_state)
    keep_idx = torch.topk(importance, target_hidden_dim).indices
    keep_idx, _ = torch.sort(keep_idx)  # arbitrary but deterministic neuron order in the smaller model

    new_model = build_head(hidden_dim=target_hidden_dim)
    with torch.no_grad():
        new_model.net[0].weight.copy_(baseline_state["net.0.weight"][keep_idx, :])
        new_model.net[0].bias.copy_(baseline_state["net.0.bias"][keep_idx])
        new_model.net[3].weight.copy_(baseline_state["net.3.weight"][:, keep_idx])
        new_model.net[3].bias.copy_(baseline_state["net.3.bias"])
    return new_model, keep_idx


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_split_data()
    baseline_state = torch.load("model/classifier_head.pt", map_location="cpu")
    baseline_params = HIDDEN_DIM * (1024 + 1 + 4) + 4

    summary = {"method": "structured_neuron_pruning", "hidden_dim_baseline": HIDDEN_DIM, "levels": []}

    for frac in NEURON_PRUNE_FRACTIONS:
        target_hidden_dim = max(1, round(HIDDEN_DIM * (1 - frac)))
        print(f"\n=== target neuron-prune fraction {frac:.0%} (hidden_dim {HIDDEN_DIM} -> {target_hidden_dim}) ===")

        model, keep_idx = build_structurally_pruned_head(baseline_state, target_hidden_dim)
        new_params = target_hidden_dim * (1024 + 1 + 4) + 4
        actual_param_reduction = 1 - new_params / baseline_params
        print(f"parameters: {baseline_params:,} -> {new_params:,} "
              f"({actual_param_reduction:.2%} real parameter reduction)")

        pre_ft = evaluate(model, X_test, y_test)
        print(f"pre-fine-tune:  accuracy={pre_ft['accuracy']:.4f} macro_f1={pre_ft['macro_f1']:.4f}")

        model, curve = fine_tune_with_mask(model, X_train, y_train, X_val, y_val, X_test, y_test, masks=None)

        val = evaluate(model, X_val, y_val)

        row = full_benchmark_row(
            model, X_test, y_test, name=f"structured_pruned_{int(frac * 100)}pct",
            extra={"target_neuron_prune_fraction": frac, "hidden_dim": target_hidden_dim,
                   "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"], "n_val": len(y_val),
                   "actual_parameter_reduction_pct": round(100 * actual_param_reduction, 2),
                   "pre_finetune_accuracy": pre_ft["accuracy"], "pre_finetune_macro_f1": pre_ft["macro_f1"]},
        )
        print(f"post-fine-tune: accuracy={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f} "
              f"(recovery: {row['accuracy'] - pre_ft['accuracy']:+.4f}) "
              f"latency_mean={row['latency_mean_ms']:.4f}ms")

        ckpt_path = CHECKPOINT_DIR / f"structured_pruned_{int(frac * 100)}pct.pt"
        torch.save({"state_dict": model.state_dict(), "hidden_dim": target_hidden_dim}, ckpt_path)

        summary["levels"].append({"target_neuron_prune_fraction": frac, "row": row, "recovery_curve": curve})

    save_json(summary, RESULTS_DIR / "structured_pruning_summary.json")


if __name__ == "__main__":
    main()
