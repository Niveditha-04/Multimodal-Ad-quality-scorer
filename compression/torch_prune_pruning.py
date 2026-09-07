"""Step 3: the same global unstructured magnitude pruning sweep as
compression/manual_magnitude_pruning.py, but computed with
torch.nn.utils.prune (prune.global_unstructured + prune.L1Unstructured)
instead of hand-written masking logic. Same sparsity levels, same target
parameters (net.0.weight, net.3.weight), same baseline checkpoint, so the
two are directly comparable.

To isolate exactly one variable -- how the mask is computed -- both
methods are then fine-tuned through the identical
compression/common.fine_tune_with_mask code path. torch.nn.utils.prune's
own reparametrization (weight = weight_orig * weight_mask, recomputed
every forward) would actually keep the model sparse through fine-tuning
without any explicit mask reapplication, which is a real, worth-noting
difference from the hand-written version -- but using it here would also
change two things at once (mask computation AND fine-tuning mechanics),
muddying the comparison. So the toolkit's mask is extracted via
prune.remove() (baking pruned weights into a plain state_dict) and handed
to the same fine-tuning function used in Step 2. That difference in how
torch.prune keeps sparsity during ITS native fine-tuning workflow is
documented in compression/README.md rather than silently absorbed.

Also checks, at every sparsity level, whether the two methods prune the
exact same weight positions -- if the math is right, they should.
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
    # bake the mask into a plain weight tensor and remove the
    # weight_orig/weight_mask reparametrization, so the model is a normal
    # MLPHead again and can go through the same fine-tuning + benchmarking
    # code as every other checkpoint in this study
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
        print(f"mask exact match vs. self-implemented (Step 2): {agree} ({n_diff} differing elements)")
        mask_agreement.append({"target_sparsity": sparsity, "masks_identical": agree, "n_differing_elements": n_diff})

        pre_ft = evaluate(model, X_test, y_test)
        print(f"pre-fine-tune:  accuracy={pre_ft['accuracy']:.4f} macro_f1={pre_ft['macro_f1']:.4f}")

        model, curve = fine_tune_with_mask(model, X_train, y_train, X_val, y_val, X_test, y_test, masks=torch_masks)

        row = full_benchmark_row(
            model, X_test, y_test, name=f"torch_prune_{int(sparsity * 100)}pct",
            extra={"target_sparsity": sparsity, "pre_finetune_accuracy": pre_ft["accuracy"],
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
