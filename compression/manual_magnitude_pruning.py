"""Unstructured global magnitude pruning, implemented without a pruning
library, on the classifier head's two Linear weight matrices (net.0.weight and
net.3.weight). Biases are left unpruned because they are a tiny share of the
parameters.

One magnitude threshold is computed across both matrices combined, everything
at or below it is zeroed through a 0/1 keep-mask, and the mask is reapplied
after every fine-tuning step (fine_tune_with_mask in compression/common.py) so
pruned weights stay at zero. torch_prune_pruning.py runs the same sweep with
torch.nn.utils.prune for comparison.

Each target sparsity is pruned one-shot from the baseline checkpoint,
evaluated immediately, then fine-tuned with the mask enforced, which gives an
accuracy-recovery curve instead of a single before/after number.
"""
import torch

from compression.common import (
    CHECKPOINT_DIR, RESULTS_DIR, build_head, evaluate, fine_tune_with_mask,
    full_benchmark_row, load_split_data, save_json,
)

PRUNABLE_PARAMS = ["net.0.weight", "net.3.weight"]
SPARSITY_LEVELS = [0.3, 0.5, 0.7, 0.9]


def compute_global_magnitude_masks(state_dict: dict, sparsity: float) -> dict:
    """Hand-written global unstructured magnitude pruning. Pools the
    absolute values of every element across ALL of PRUNABLE_PARAMS into one
    ranking (global, not per-layer), finds the value at the `sparsity`
    percentile via torch.kthvalue, and returns one 0/1 mask per parameter
    marking which elements survive (1) vs. get pruned (0).
    """
    all_abs = torch.cat([state_dict[name].abs().flatten() for name in PRUNABLE_PARAMS])
    n_total = all_abs.numel()
    n_prune = int(round(sparsity * n_total))

    if n_prune <= 0:
        threshold = -1.0  # keep everything
    elif n_prune >= n_total:
        threshold = float("inf")  # prune everything
    else:
        # the n_prune-th smallest magnitude is the cutoff; everything at or
        # below it is pruned, everything strictly above survives
        threshold = torch.kthvalue(all_abs, n_prune).values.item()

    masks = {}
    for name in PRUNABLE_PARAMS:
        w = state_dict[name]
        masks[name] = (w.abs() > threshold).float()
    return masks


def apply_masks_inplace(model: torch.nn.Module, masks: dict):
    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, mask in masks.items():
            named_params[name].mul_(mask)


def actual_sparsity(masks: dict) -> float:
    total = sum(m.numel() for m in masks.values())
    zeros = sum(int((m == 0).sum().item()) for m in masks.values())
    return zeros / total


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_split_data()
    baseline_state = torch.load("model/classifier_head.pt", map_location="cpu")

    summary = {"method": "self_implemented_magnitude_pruning", "prunable_params": PRUNABLE_PARAMS, "levels": []}

    for sparsity in SPARSITY_LEVELS:
        print(f"\n=== target sparsity {sparsity:.0%} ===")
        model = build_head()
        model.load_state_dict(baseline_state)

        masks = compute_global_magnitude_masks(baseline_state, sparsity)
        apply_masks_inplace(model, masks)
        achieved_sparsity = actual_sparsity(masks)
        print(f"achieved sparsity on prunable weights: {achieved_sparsity:.4f}")

        pre_ft = evaluate(model, X_test, y_test)
        print(f"pre-fine-tune:  accuracy={pre_ft['accuracy']:.4f} macro_f1={pre_ft['macro_f1']:.4f}")

        model, curve = fine_tune_with_mask(model, X_train, y_train, X_val, y_val, X_test, y_test, masks=masks)

        val = evaluate(model, X_val, y_val)

        row = full_benchmark_row(
            model, X_test, y_test, name=f"self_pruned_{int(sparsity * 100)}pct",
            extra={"target_sparsity": sparsity, "achieved_sparsity_on_prunable_weights": achieved_sparsity,
                   "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"], "n_val": len(y_val),
                   "pre_finetune_accuracy": pre_ft["accuracy"], "pre_finetune_macro_f1": pre_ft["macro_f1"]},
        )
        print(f"post-fine-tune: accuracy={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f} "
              f"(recovery: {row['accuracy'] - pre_ft['accuracy']:+.4f})")

        ckpt_path = CHECKPOINT_DIR / f"self_pruned_{int(sparsity * 100)}pct.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

        summary["levels"].append({"target_sparsity": sparsity, "row": row, "recovery_curve": curve})

    save_json(summary, RESULTS_DIR / "self_pruning_summary.json")


if __name__ == "__main__":
    main()
