"""INT8 post-training quantization of the checkpoint ranked first by
compression/select_best_checkpoint.py (best validation accuracy among the
pruned checkpoints, ties broken by gzip size).

Uses torch.quantization.quantize_dynamic. The model is two Linear layers and a
ReLU with nothing to fuse, so the usual advantages of static quantization
(operator fusion, calibrated activation ranges) do not apply, and dynamic
quantization needs no calibration data. Weights are quantized once at
conversion time and activations on every forward call.

Needs a quantized-engine backend: fbgemm when available (typical on x86),
otherwise qnnpack (the only one on Apple Silicon).
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

    # Self-describing checkpoint: structured_pruning.py's {"state_dict",
    # "hidden_dim"} format plus "quantized" and "quantization_engine". A loader
    # can rebuild the right module shape and quantize it before load_state_dict
    # without any extra configuration.
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
