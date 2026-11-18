"""The global unstructured magnitude pruning sweep of
manual_magnitude_pruning.py, computed with torch.nn.utils.prune
(global_unstructured with L1Unstructured) instead of hand-written masking.
Same sparsity levels, target parameters (net.0.weight, net.3.weight) and
baseline checkpoint.

Only the mask computation differs. torch.nn.utils.prune's reparametrization
(weight = weight_orig * weight_mask) would keep the model sparse during
fine-tuning without reapplying a mask, but using it would change the
fine-tuning mechanics as well. So the toolkit's mask is extracted, baked into
a plain state_dict with prune.remove(), and fine-tuned with the same
fine_tune_with_mask function as the hand-written version.

At every sparsity level the script also checks that the two methods prune
exactly the same weight positions.
"""
import torch
import torch.nn.utils.prune as prune

from compression.common import (
    CHECKPOINT_DIR, RESULTS_DIR, build_head, evaluate, fine_tune_with_mask,
    full_benchmark_row, load_split_data, save_json,
)
from compression.manual_magnitude_pruning import PRUNABLE_PARAMS, SPARSITY_LEVELS, compute_global_magnitude_masks


def torch_prune_masks(model: torch.nn.Module, sparsity: float) -> dict:
    parameters_to_prune = [(model.net[0], "weight"), (model.net[3], "weight")]
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=sparsity)

    masks = {
        "net.0.weight": model.net[0].weight_mask.clone(),
        "net.3.weight": model.net[3].weight_mask.clone(),
    }
    # Bake the mask into a plain weight tensor and remove the
    # weight_orig/weight_mask reparametrization, so the model is a normal
    # MLPHead again for the shared fine-tuning and benchmarking code.
    prune.remove(model.net[0], "weight")
    prune.remove(model.net[3], "weight")
    return masks


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_split_data()
    baseline_state = torch.load("model/classifier_head.pt", map_location="cpu")

    summary = {"method": "torch_nn_utils_prune", "prunable_params": PRUNABLE_PARAMS, "levels": []}
    mask_agreement = []

    for sparsity in SPARSITY_LEVELS:
        print(f"\n=== target sparsity {sparsity:.0%} (torch.nn.utils.prune) ===")
        model = build_head()
        model.load_state_dict(baseline_state)

        torch_masks = torch_prune_masks(model, sparsity)

        self_masks = compute_global_magnitude_masks(baseline_state, sparsity)
        agree = all(
            torch.equal(torch_masks[name], self_masks[name]) for name in PRUNABLE_PARAMS
        )
        n_diff = sum(
            int((torch_masks[name] != self_masks[name]).sum().item()) for name in PRUNABLE_PARAMS
        )
        print(f"mask exact match vs. self-implemented: {agree} ({n_diff} differing elements)")
        mask_agreement.append({"target_sparsity": sparsity, "masks_identical": agree, "n_differing_elements": n_diff})

        pre_ft = evaluate(model, X_test, y_test)
        print(f"pre-fine-tune:  accuracy={pre_ft['accuracy']:.4f} macro_f1={pre_ft['macro_f1']:.4f}")

        model, curve = fine_tune_with_mask(model, X_train, y_train, X_val, y_val, X_test, y_test, masks=torch_masks)

        val = evaluate(model, X_val, y_val)

        row = full_benchmark_row(
            model, X_test, y_test, name=f"torch_prune_{int(sparsity * 100)}pct",
            extra={"target_sparsity": sparsity, "pre_finetune_accuracy": pre_ft["accuracy"],
                   "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"], "n_val": len(y_val),
                   "pre_finetune_macro_f1": pre_ft["macro_f1"], "masks_identical_to_self_implemented": agree},
        )
        print(f"post-fine-tune: accuracy={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f} "
              f"(recovery: {row['accuracy'] - pre_ft['accuracy']:+.4f})")

        ckpt_path = CHECKPOINT_DIR / f"torch_prune_{int(sparsity * 100)}pct.pt"
        torch.save(model.state_dict(), ckpt_path)

        summary["levels"].append({"target_sparsity": sparsity, "row": row, "recovery_curve": curve})

    summary["mask_agreement_with_self_implemented"] = mask_agreement
    save_json(summary, RESULTS_DIR / "torch_prune_summary.json")

    print("\n=== mask agreement summary ===")
    for m in mask_agreement:
        print(f"  sparsity={m['target_sparsity']:.0%} identical={m['masks_identical']} diff_elements={m['n_differing_elements']}")


if __name__ == "__main__":
    main()
