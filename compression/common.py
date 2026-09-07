"""Shared measurement and I/O utilities for the classifier-head compression
study. CLIP's backbone is out of scope everywhere in this module -- every
function here operates on the small MLPHead only, using embeddings already
extracted to model/embeddings.npz by model/extract_embeddings.py.

Reuses model/train_classifier.py's exact train/val/test split logic so
every number produced here is directly comparable to model/results/
metrics.json (the original baseline), not a re-derived approximation.
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
    """Size of the dense state_dict serialized with torch.save, in bytes.
    This is what actually lands on disk / in a Docker image for a model
    that isn't using a sparse tensor format -- so unstructured-pruned
    models with many zeros do NOT shrink here. That is a real, honest
    finding of this study, not a bug in the measurement -- see
    compression/README.md.
    """
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


def state_dict_gzip_size_bytes(model: torch.nn.Module) -> int:
    """Size of the same state_dict after gzip compression. Zeros compress
    extremely well, so this gives a realistic 'achievable on-disk size'
    for unstructured-pruned checkpoints without needing a sparse tensor
    format -- a fair, honest way to credit unstructured pruning for real
    compression potential instead of reporting 0% for every unstructured
    result.
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


def full_benchmark_row(model: torch.nn.Module, X_test: np.ndarray, y_test: np.ndarray, name: str, extra: dict | None = None) -> dict:
    """One row of the final comparison table: correctness + size + speed,
    all measured directly against the actual model object passed in (never
    hand-typed), for the given test set.
    """
    metrics = evaluate(model, X_test, y_test)
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
