"""Trains a 2-layer MLP classifier head on top of frozen CLIP embeddings.

CLIP's backbone is never touched here -- embeddings were already extracted
by extract_embeddings.py and saved to disk. This script only trains the
small head on top, in one of three modality configurations:

  image  - image embedding only (512-dim)
  text   - text embedding only (512-dim)
  joint  - concatenated image+text embeddings (1024-dim)

All three load the exact same train/test split from data/split.json, so
their metrics are directly comparable -- that comparison is the point (see
the ablation writeup in results/ablation.json). Embeddings are L2-normalized
per modality before use/concatenation, since raw CLIP image and text
embeddings have different norms (~11 vs ~8) and cosine similarity -- what
CLIP was actually trained on -- implies unit-normalized vectors.

The test set is touched exactly once, at the very end, for final metrics.
Early stopping uses a validation split carved out of the train set only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
)
from sklearn.model_selection import train_test_split

from model.clip_features import l2_normalize, build_joint_features
from model.classifier_architecture import LABELS, LABEL_TO_IDX, HIDDEN_DIM, DROPOUT, MLPHead

RANDOM_SEED = 42
VAL_FRACTION_OF_TRAIN = 0.15
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 200
PATIENCE = 20


def load_features(modality: str):
    d = np.load("model/embeddings.npz", allow_pickle=True)
    ad_ids = d["ad_ids"]
    labels = d["labels"]
    raw_image_embeds = d["image_embeds"]
    raw_text_embeds = d["text_embeds"]

    if modality == "image":
        features = l2_normalize(raw_image_embeds)
    elif modality == "text":
        features = l2_normalize(raw_text_embeds)
    elif modality == "joint":
        features = build_joint_features(raw_image_embeds, raw_text_embeds)
    else:
        raise ValueError(modality)

    return ad_ids, features, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["image", "text", "joint"], required=True)
    args = parser.parse_args()

    split = json.loads(Path("data/split.json").read_text())
    train_ids_all = set(split["train_ad_ids"])
    test_ids_all = set(split["test_ad_ids"])

    ad_ids, features, labels_str = load_features(args.modality)
    y = np.array([LABEL_TO_IDX[l] for l in labels_str])

    train_mask = np.array([i in train_ids_all for i in ad_ids])
    test_mask = np.array([i in test_ids_all for i in ad_ids])
    assert train_mask.sum() == len(train_ids_all), "train id mismatch between split.json and embeddings"
    assert test_mask.sum() == len(test_ids_all), "test id mismatch between split.json and embeddings"
    assert not (train_mask & test_mask).any()

    X_train_full, y_train_full = features[train_mask], y[train_mask]
    X_test, y_test = features[test_mask], y[test_mask]

    # carve validation set out of train only, for early stopping -- test set
    # is not touched until final eval below
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VAL_FRACTION_OF_TRAIN,
        stratify=y_train_full, random_state=RANDOM_SEED,
    )
    print(f"modality={args.modality} | train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cpu")  # tiny MLP, CPU is plenty and keeps this deterministic

    model = MLPHead(input_dim=X_train.shape[1], hidden_dim=HIDDEN_DIM, n_classes=len(LABELS), dropout=DROPOUT)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

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

        if epoch % 20 == 0 or epoch == MAX_EPOCHS - 1:
            print(f"  epoch {epoch:3d} | train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

        if epochs_no_improve >= PATIENCE:
            print(f"  early stopping at epoch {epoch}, best val_loss={best_val_loss:.4f}")
            break

    model.load_state_dict(best_state)
    model.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_probs = torch.softmax(test_logits, dim=1).numpy()
        test_preds = test_logits.argmax(dim=1).numpy()

    acc = accuracy_score(y_test, test_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, test_preds, labels=list(range(len(LABELS))), zero_division=0,
    )
    cm = confusion_matrix(y_test, test_preds, labels=list(range(len(LABELS))))

    print(f"\ntest accuracy: {acc:.4f}")
    for i, label in enumerate(LABELS):
        print(f"  {label:18s} precision={precision[i]:.3f} recall={recall[i]:.3f} f1={f1[i]:.3f} support={support[i]}")
    print("confusion matrix (rows=true, cols=pred):")
    print(LABELS)
    print(cm)

    results = {
        "modality": args.modality,
        "input_dim": X_train.shape[1],
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "accuracy": acc,
        "per_class": {
            LABELS[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            } for i in range(len(LABELS))
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": LABELS,
        "best_val_loss": best_val_loss,
    }

    Path("model/results").mkdir(parents=True, exist_ok=True)
    out_path = f"model/results/{args.modality}_results.json"
    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"\nsaved results to {out_path}")

    if args.modality == "joint":
        torch.save(model.state_dict(), "model/classifier_head.pt")
        print("saved trained joint classifier weights to model/classifier_head.pt")


if __name__ == "__main__":
    main()
