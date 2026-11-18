"""Shared measurement and I/O utilities for the classifier-head compression
study. Everything here operates on the small MLPHead only, using embeddings
already extracted to model/embeddings.npz by model/extract_embeddings.py. The
CLIP backbone is not involved.

The train/val/test split logic comes from model/train_classifier.py, so the
numbers are directly comparable to model/results/joint_results.json.
"""
import gzip
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from model.classifier_architecture import DROPOUT, HIDDEN_DIM, LABEL_TO_IDX, LABELS, MLPHead
from model.train_classifier import RANDOM_SEED, VAL_FRACTION_OF_TRAIN, load_features

RESULTS_DIR = Path("compression/results")
CHECKPOINT_DIR = Path("compression/checkpoints")


def load_split_data():
    """Returns (X_train, y_train, X_val, y_val, X_test, y_test) for the
    joint modality, using the identical split and val carve-out as
    model/train_classifier.py --modality joint. Deterministic: same
    RANDOM_SEED, same stratify call, same order of operations.
    """
    split = json.loads(Path("data/split.json").read_text())
    train_ids_all = set(split["train_ad_ids"])
    test_ids_all = set(split["test_ad_ids"])

    ad_ids, features, labels_str = load_features("joint")
    y = np.array([LABEL_TO_IDX[l] for l in labels_str])

    train_mask = np.array([i in train_ids_all for i in ad_ids])
    test_mask = np.array([i in test_ids_all for i in ad_ids])
    assert train_mask.sum() == len(train_ids_all)
    assert test_mask.sum() == len(test_ids_all)
    assert not (train_mask & test_mask).any()

    X_train_full, y_train_full = features[train_mask], y[train_mask]
    X_test, y_test = features[test_mask], y[test_mask]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VAL_FRACTION_OF_TRAIN,
        stratify=y_train_full, random_state=RANDOM_SEED,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_head(input_dim: int = 1024, hidden_dim: int = HIDDEN_DIM) -> MLPHead:
    return MLPHead(input_dim=input_dim, hidden_dim=hidden_dim, n_classes=len(LABELS), dropout=DROPOUT)


def load_checkpoint(path) -> torch.nn.Module:
    """Loads any checkpoint format used here: a plain state_dict (the
    unstructured-pruned checkpoints keep the baseline's hidden_dim of 128) or
    the self-describing {"state_dict", "hidden_dim", "quantized"} dict saved by
    structured_pruning.py and quantize_int8.py. A structurally pruned model has
    a non-default hidden_dim, and a quantized model must be converted with
    quantize_dynamic() before load_state_dict, because its state_dict holds
    packed int8 parameters. api/main.py uses the same logic.
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and "hidden_dim" in obj:
        model = build_head(hidden_dim=obj["hidden_dim"])
        if obj.get("quantized"):
            engine = obj.get("quantization_engine", "qnnpack")
            if engine not in torch.backends.quantized.supported_engines:
                engine = torch.backends.quantized.supported_engines[0]
            torch.backends.quantized.engine = engine
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        model.load_state_dict(obj["state_dict"])
    else:
        model = build_head()
        model.load_state_dict(obj)
    model.eval()
    return model


def load_baseline_head() -> MLPHead:
    model = build_head()
    model.load_state_dict(torch.load("model/classifier_head.pt", map_location="cpu"))
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: torch.nn.Module, X: np.ndarray, y: np.ndarray) -> dict:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    logits = model(X_t)
    preds = logits.argmax(dim=1).numpy()

    acc = accuracy_score(y, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, preds, labels=list(range(len(LABELS))), zero_division=0,
    )
    macro_f1 = float(np.mean(f1))

    return {
        "accuracy": float(acc),
        "macro_f1": macro_f1,
        "per_class": {
            LABELS[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            } for i in range(len(LABELS))
        },
    }


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model: torch.nn.Module) -> int:
    total = 0
    for p in model.parameters():
        total += int(torch.count_nonzero(p).item())
    return total


def state_dict_raw_size_bytes(model: torch.nn.Module) -> int:
    """Size of the dense state_dict serialized with torch.save, in bytes. This
    is the size on disk or in a Docker image when no sparse format is used, so
    unstructured-pruned models with many zeros do not shrink.
    """
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


def state_dict_gzip_size_bytes(model: torch.nn.Module) -> int:
    """Size of the same state_dict after gzip compression. Zeros compress very
    well, so this approximates the achievable on-disk size of unstructured-pruned
    checkpoints without needing a sparse tensor format.
    """
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(gzip.compress(buf.getvalue(), compresslevel=9))


@torch.no_grad()
def measure_latency_ms(model: torch.nn.Module, input_dim: int = 1024, n_warmup: int = 50, n_trials: int = 500) -> dict:
    """Single-sample (batch_size=1) CPU inference latency, matching how the
    FastAPI /score endpoint actually calls the head: one ad at a time, no
    explicit device placement (the classifier head runs on CPU in
    api/main.py regardless of where CLIP itself runs). Reports mean,
    median, and p95 in milliseconds over n_trials timed calls, after
    n_warmup untimed calls to avoid first-call overhead skewing the result.
    """
    model.eval()
    x = torch.randn(1, input_dim, dtype=torch.float32)

    for _ in range(n_warmup):
        model(x)

    times_ms = []
    for _ in range(n_trials):
        start = time.perf_counter()
        model(x)
        times_ms.append((time.perf_counter() - start) * 1000)

    arr = np.array(times_ms)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "n_trials": n_trials,
    }


def full_benchmark_row(
    model: torch.nn.Module, X_test: np.ndarray, y_test: np.ndarray, name: str,
    extra: dict | None = None, parameter_counts_override: tuple[int, int] | None = None,
) -> dict:
    """One row of the benchmark table: accuracy, size and speed, all measured
    from the model object passed in.

    parameter_counts_override is (total, nonzero) and is only for dynamically
    quantized models. quantize_dynamic stores weights in a packed structure
    that is not an nn.Parameter, so model.parameters() is empty and
    count_parameters would return 0. Quantization changes each weight's bit
    width, not the number of weights, so the override is the pre-quantization
    parameter count.
    """
    metrics = evaluate(model, X_test, y_test)
    if parameter_counts_override is not None:
        total_params, nonzero_params = parameter_counts_override
    else:
        total_params = count_parameters(model)
        nonzero_params = count_nonzero_parameters(model)
    raw_size = state_dict_raw_size_bytes(model)
    gzip_size = state_dict_gzip_size_bytes(model)
    latency = measure_latency_ms(model)

    row = {
        "name": name,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class": metrics["per_class"],
        "total_parameters": total_params,
        "nonzero_parameters": nonzero_params,
        "sparsity_pct": round(100.0 * (1 - nonzero_params / total_params), 2),
        "raw_size_bytes": raw_size,
        "gzip_size_bytes": gzip_size,
        "latency_mean_ms": latency["mean_ms"],
        "latency_median_ms": latency["median_ms"],
        "latency_p95_ms": latency["p95_ms"],
    }
    if extra:
        row.update(extra)
    return row


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    print(f"saved {path}")


FT_LR = 1e-3
FT_WEIGHT_DECAY = 1e-4
FT_MAX_EPOCHS = 60
FT_PATIENCE = 15


def fine_tune_with_mask(
    model,
    X_train, y_train, X_val, y_val, X_test, y_test,
    masks: dict | None = None,
    max_epochs: int = FT_MAX_EPOCHS,
    patience: int = FT_PATIENCE,
    lr: float = FT_LR,
    weight_decay: float = FT_WEIGHT_DECAY,
    log_every: int = 5,
):
    """Fine-tunes `model` in place after pruning. If `masks` is given (a dict of
    param_name -> 0/1 tensor with the parameter's shape), the mask is reapplied
    after every optimizer step so pruned weights cannot be revived by gradient
    updates. Structured pruning needs no masks, since the pruned neurons are
    removed from the model.

    The best checkpoint is chosen by validation loss only, as in
    model/train_classifier.py. Test accuracy is evaluated every `log_every`
    epochs to build the accuracy-recovery curve. It is never used for early
    stopping or checkpoint selection.

    Seeds with RANDOM_SEED before training. The head uses nn.Dropout(0.3),
    which samples a new mask on every training forward pass, so without a seed
    two runs from an identical model and prune mask would diverge. Seeding
    makes the same mask always give the same fine-tuned result, which lets the
    self-implemented and torch.nn.utils.prune sweeps differ only in how the
    mask is computed.
    """
    torch.manual_seed(RANDOM_SEED)
    named_params = dict(model.named_parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    def apply_masks():
        if not masks:
            return
        with torch.no_grad():
            for name, mask in masks.items():
                named_params[name].mul_(mask)

    apply_masks()  # in case the incoming model wasn't already masked

    best_val_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    epochs_no_improve = 0
    recovery_curve = []

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()
        apply_masks()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % log_every == 0 or epoch == max_epochs - 1:
            test_metrics = evaluate(model, X_test, y_test)
            recovery_curve.append({
                "epoch": epoch,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            })

        if epochs_no_improve >= patience:
            break

    model.load_state_dict(best_state)
    apply_masks()
    model.eval()
    return model, recovery_curve
