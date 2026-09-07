"""Step 6: INT8 post-training quantization, applied to whichever checkpoint
compression/select_best_checkpoint.py picked (structured pruning at 75%
neuron removal, selected on accuracy -- see that script's output).

Uses torch.quantization.quantize_dynamic rather than static PTQ. This
model is two nn.Linear layers with a ReLU in between and no
convolution/batchnorm to fuse, so the usual reason to reach for static
quantization (fusing conv+bn+relu, then calibrating activation ranges with
a representative dataset) doesn't apply here -- there's nothing to fuse,
and dynamic quantization already gives INT8 weights with no calibration
data required, which is simpler and just as appropriate at this model
size. Static quantization would still be a reasonable choice; dynamic is
not a shortcut taken to avoid work, it is the better-fit tool for a
Linear-only network this small. This tradeoff is documented, not hidden.

Weights are quantized statically (fixed at conversion time, from the
already-fine-tuned float checkpoint); activations are quantized
dynamically per forward call. Requires a quantized-engine backend --
this machine only has qnnpack available (Apple Silicon); fbgemm is
preferred when present (typical on x86), matching PyTorch's own default
preference order.
"""
import json

import torch

from compression.common import (
    CHECKPOINT_DIR, RESULTS_DIR, count_parameters, evaluate, full_benchmark_row,
    load_checkpoint, load_split_data, save_json,
)


def pick_quantization_engine() -> str:
    supported = torch.backends.quantized.supported_engines
    for preferred in ("fbgemm", "qnnpack"):
        if preferred in supported:
            return preferred
    raise RuntimeError(f"no usable quantized engine found; supported={supported}")


def main():
    selection = json.loads((RESULTS_DIR / "best_checkpoint_selection.json").read_text())["selected"]
    print(f"quantizing selected checkpoint: {selection['method']} @ {selection['level_pct']}% "
          f"({selection['checkpoint_path']})")

    engine = pick_quantization_engine()
    torch.backends.quantized.engine = engine
    print(f"quantized engine backend: {engine}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_split_data()

    float_model = load_checkpoint(selection["checkpoint_path"])
    float_total_params = count_parameters(float_model)

    float_metrics = evaluate(float_model, X_test, y_test)
    print(f"\nfloat32 fine-tuned checkpoint (re-verified here): "
          f"accuracy={float_metrics['accuracy']:.4f} macro_f1={float_metrics['macro_f1']:.4f}")
    expected = selection["accuracy"]
    if abs(float_metrics["accuracy"] - expected) > 1e-9:
        raise RuntimeError(
            f"loaded checkpoint's accuracy ({float_metrics['accuracy']:.6f}) doesn't match the "
            f"selection record ({expected:.6f}) -- wrong checkpoint or state drifted, stop and investigate."
        )

    float_row = full_benchmark_row(float_model, X_test, y_test, name="best_pruned_finetuned_fp32")

    quantized_model = torch.quantization.quantize_dynamic(
        float_model, {torch.nn.Linear}, dtype=torch.qint8,
    )
    quantized_model.eval()

    quant_metrics = evaluate(quantized_model, X_test, y_test)
    print(f"\nINT8 dynamically-quantized: accuracy={quant_metrics['accuracy']:.4f} "
          f"macro_f1={quant_metrics['macro_f1']:.4f}")
    print(f"accuracy delta from quantization alone: {quant_metrics['accuracy'] - float_metrics['accuracy']:+.4f}")

    quant_row = full_benchmark_row(
        quantized_model, X_test, y_test, name="best_pruned_finetuned_int8",
        parameter_counts_override=(float_total_params, float_total_params),
        extra={"quantization_engine": engine, "source_checkpoint": selection["checkpoint_path"],
               "source_method": selection["method"], "source_level_pct": selection["level_pct"]},
    )

    print(f"\nsize:    fp32 raw={float_row['raw_size_bytes']:,}B gzip={float_row['gzip_size_bytes']:,}B  "
          f"->  int8 raw={quant_row['raw_size_bytes']:,}B gzip={quant_row['gzip_size_bytes']:,}B")
    print(f"latency: fp32 mean={float_row['latency_mean_ms']:.4f}ms  ->  int8 mean={quant_row['latency_mean_ms']:.4f}ms")
    size_reduction = 1 - quant_row["raw_size_bytes"] / float_row["raw_size_bytes"]
    print(f"raw size reduction from quantization alone: {size_reduction:.2%}")

    # self-describing checkpoint, same convention as structured_pruning.py's
    # {"state_dict", "hidden_dim"} format plus "quantized"/"quantization_engine"
    # -- so anything loading this checkpoint (the API included) can
    # reconstruct the right module shape and quantize it BEFORE calling
    # load_state_dict, without needing to be told out-of-band. A plain
    # state_dict alone can't self-describe "this needs hidden_dim=32 and a
    # quantize_dynamic() call before loading", and getting that wrong from a
    # manually-set config value is exactly the kind of silent failure mode
    # this project has caught and fixed elsewhere (see db/session.py's
    # DATABASE_URL bug on the v2-agentic-eval branch).
    torch.save(
        {"state_dict": quantized_model.state_dict(), "hidden_dim": float_model.net[0].out_features,
         "quantized": True, "quantization_engine": engine},
        CHECKPOINT_DIR / "best_int8_quantized.pt",
    )

    save_json(
        {"selected_source": selection, "float_row": float_row, "quantized_row": quant_row,
         "quantization_engine": engine, "raw_size_reduction_from_quantization_alone_pct": round(100 * size_reduction, 2)},
        RESULTS_DIR / "quantization_summary.json",
    )


if __name__ == "__main__":
    main()
