"""Generates the figures used in README.md, all from saved files and with no
API calls:

  docs/images/confusion_matrix.png       model/results/joint_results.json
  docs/images/ablation_comparison.png    model/results/ablation_summary.json
  docs/images/compression_accuracy.png   compression/results/final_benchmark.json
                                         and best_checkpoint_selection.json
  docs/images/example_predictions.png    test predictions of the baseline head,
                                         data/dataset.csv, data/images/ and
                                         eval/generated_explanations.json

The confusion matrix uses a single-hue sequential blue ramp (magnitude). The
other charts use the categorical colors below. example_predictions.png needs
the photos, which `python data/build_dataset.py` rebuilds into data/images/.
"""
import json
import re
import textwrap
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRAY = "#8a8985"
GREEN = "#1b7f3b"
RED = "#b3261e"
N_TEST = 96


def make_confusion_matrix():
    metrics = json.load(open("model/results/joint_results.json"))
    cm = np.array(metrics["confusion_matrix"])
    labels = metrics["confusion_matrix_labels"]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    # sequential blue ramp, light to dark, for a magnitude encoding
    cmap = plt.cm.Blues
    im = ax.imshow(cm, cmap=cmap, vmin=0)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", color=TEXT_PRIMARY)
    ax.set_yticklabels(labels, color=TEXT_PRIMARY)
    ax.set_xlabel("Predicted label", color=TEXT_PRIMARY, fontsize=11)
    ax.set_ylabel("True label", color=TEXT_PRIMARY, fontsize=11)
    ax.set_title(
        f"Joint classifier confusion matrix (test set, n=96)\n"
        f"accuracy = {metrics['accuracy']:.3f}",
        color=TEXT_PRIMARY, fontsize=12,
    )

    max_val = cm.max()
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = cm[i, j]
            text_color = "white" if value > max_val * 0.55 else TEXT_PRIMARY
            ax.text(j, i, str(value), ha="center", va="center", color=text_color, fontsize=13)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    plt.tight_layout()
    plt.savefig("docs/images/confusion_matrix.png", dpi=150, facecolor="white")
    plt.close()
    print("wrote docs/images/confusion_matrix.png")


def make_ablation_comparison():
    ablation = json.load(open("model/results/ablation_summary.json"))
    classes = ["approved", "policy_violation", "misleading", "low_quality"]
    modalities = ["image_only", "text_only", "joint"]
    colors = {"image_only": BLUE, "text_only": ORANGE, "joint": AQUA}
    display_names = {"image_only": "Image only", "text_only": "Text only", "joint": "Joint (image + text)"}

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(classes))
    width = 0.26

    for i, modality in enumerate(modalities):
        values = [ablation["recall_by_class"][c][modality] for c in classes]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=display_names[modality],
                      color=colors[modality], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, color=TEXT_SECONDARY)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall", color=TEXT_PRIMARY, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in classes], color=TEXT_PRIMARY)
    ax.set_title(
        f"Per-class recall by modality (identical train/test split)\n"
        f"Overall accuracy: image {ablation['accuracy']['image_only']:.3f}, "
        f"text {ablation['accuracy']['text_only']:.3f}, "
        f"joint {ablation['accuracy']['joint']:.3f}",
        color=TEXT_PRIMARY, fontsize=11,
    )
    ax.legend(frameon=False, loc="upper left", labelcolor=TEXT_PRIMARY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d9d8d3")
    ax.spines["bottom"].set_color("#d9d8d3")
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.yaxis.grid(True, color="#ececea", linewidth=0.8)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig("docs/images/ablation_comparison.png", dpi=150, facecolor="white")
    plt.close()
    print("wrote docs/images/ablation_comparison.png")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def make_compression_accuracy():
    """File size vs test accuracy, with 95% Wilson intervals (n = 96 test ads).
    X is the dense state_dict size, so unstructured pruning stays at the
    baseline's size while structured pruning and INT8 move left.
    """
    rows = {r["label"]: r for r in json.load(open("compression/results/final_benchmark.json"))["rows"]}
    selection = json.load(open("compression/results/best_checkpoint_selection.json"))
    chosen = selection["selected"]

    def label_of(method: str, pct: int) -> str:
        if method == "structured":
            return f"structured {pct}% neurons removed"
        return f"unstructured {pct}% sparsity"

    quantized = next(r for r in rows.values() if r["group"] == "quantized")
    kind = "structured" if chosen["method"] == "structured" else "unstructured"
    selected_label = label_of(chosen["method"], chosen["level_pct"])
    points = [
        ("Baseline", rows["baseline (fp32, unpruned)"], "baseline", (14, -26)),
        (f"Selected: {kind} {chosen['level_pct']}%", rows[selected_label], kind, (14, 4)),
        ("Unstructured 90%", rows["unstructured 90% sparsity"], "unstructured", (14, 16)),
        ("Structured 90%", rows["structured 90% neurons removed"], "structured", (14, -18)),
        ("Structured 75% + INT8", quantized, "int8", (14, 24)),
    ]
    if kind != "structured":
        best_structured = next(c for c in selection["all_candidates_ranked"] if c["method"] == "structured")
        points.append((f"Best structured on validation: {best_structured['level_pct']}%",
                       rows[label_of("structured", best_structured["level_pct"])], "structured", (14, 10)))
    points = [pt for i, pt in enumerate(points) if not any(q[1] is pt[1] for q in points[:i])]
    families = {"baseline": (GRAY, "s", "Baseline"), "unstructured": (BLUE, "o", "Unstructured pruning"),
                "structured": (AQUA, "o", "Structured pruning"), "int8": (ORANGE, "D", "INT8 quantized")}

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axhline(rows["baseline (fp32, unpruned)"]["accuracy"], color="#d9d8d3", linewidth=1, zorder=1)
    seen = set()
    for name, r, family, offset in points:
        color, marker, legend_label = families[family]
        k = round(r["accuracy"] * N_TEST)
        lo, hi = wilson_interval(k, N_TEST)
        x = r["raw_size_bytes"] / 1000
        ax.errorbar(x, r["accuracy"], yerr=[[r["accuracy"] - lo], [hi - r["accuracy"]]], fmt=marker, color=color,
                    markersize=8, capsize=4, linewidth=1.6, markeredgecolor="white", zorder=3,
                    label=legend_label if legend_label not in seen else None)
        seen.add(legend_label)
        ax.annotate(f"{name}\n{k}/{N_TEST} correct", (x, r["accuracy"]), textcoords="offset points", xytext=offset,
                    fontsize=8, color=TEXT_PRIMARY, va="center")

    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(0, 700)
    ax.set_xlabel("Model file size (KB, dense state_dict)", color=TEXT_PRIMARY, fontsize=10)
    ax.set_ylabel("Test accuracy", color=TEXT_PRIMARY, fontsize=10)
    ax.set_title("Compressed classifier heads: file size vs. test accuracy\n"
                 "95% Wilson intervals, n = 96 (one ad = 1.04 points)", color=TEXT_PRIMARY, fontsize=11)
    ax.legend(frameon=False, loc="lower right", labelcolor=TEXT_PRIMARY, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d9d8d3")
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.yaxis.grid(True, color="#ececea", linewidth=0.8)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("docs/images/compression_accuracy.png", dpi=150, facecolor="white")
    plt.close()
    print("wrote docs/images/compression_accuracy.png")


def test_predictions() -> list[dict]:
    """Baseline head predictions for every test ad: true label, predicted label
    and softmax confidence."""
    from model.classifier_architecture import DROPOUT, HIDDEN_DIM, LABELS, MLPHead
    from model.train_classifier import load_features

    ad_ids, features, labels = load_features("joint")
    test_ids = set(json.load(open("data/split.json"))["test_ad_ids"])
    head = MLPHead(input_dim=features.shape[1], hidden_dim=HIDDEN_DIM, n_classes=len(LABELS), dropout=DROPOUT)
    head.load_state_dict(torch.load("model/classifier_head.pt", map_location="cpu"))
    head.eval()
    with torch.no_grad():
        probs = torch.softmax(head(torch.tensor(features, dtype=torch.float32)), dim=1).numpy()
    return [
        {"ad_id": int(a), "true": str(t), "pred": LABELS[int(p.argmax())], "conf": float(p.max())}
        for a, t, p in zip(ad_ids, labels, probs) if int(a) in test_ids
    ]


def select_examples(rows: list[dict], explained: set[int]) -> list[dict]:
    """Six test ads, always the lowest ad_id in each case:
      - one correct ad per class (for the three flagged classes, the lowest
        ad_id that has a saved explanation, since no new API calls are made)
      - one wrong ad for each of the two most frequent error types
    """
    from model.classifier_architecture import LABELS

    chosen = []
    for label in LABELS:
        ok = [r for r in rows if r["true"] == label and r["pred"] == label
              and (label == "approved" or r["ad_id"] in explained)]
        chosen.append(min(ok, key=lambda r: r["ad_id"]))
    wrong = [r for r in rows if r["true"] != r["pred"]]
    for pair, _ in sorted(Counter((r["true"], r["pred"]) for r in wrong).items(), key=lambda kv: (-kv[1], kv[0]))[:2]:
        chosen.append(min((r for r in wrong if (r["true"], r["pred"]) == pair), key=lambda r: r["ad_id"]))
    return chosen


def first_sentence(text: str) -> str:
    text = re.sub(r"^#+ .*$", "", text, flags=re.M).replace("**", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*\u2014\s*", ", ", text)
    return re.split(r"(?<=[.!?][\"'])\s|(?<=[.!?])\s", text, maxsplit=1)[0]


def clipped(text: str, width: int, max_lines: int) -> str:
    lines = textwrap.wrap(text, width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .,;") + "..."
    return "\n".join(lines)


def make_example_predictions():
    df = pd.read_csv("data/dataset.csv")
    explanations = {int(e["ad_id"]): e["explanation"] for e in json.load(open("eval/generated_explanations.json"))}
    examples = select_examples(test_predictions(), set(explanations))

    fig = plt.figure(figsize=(13, 6.9))
    outer = fig.add_gridspec(2, 3, hspace=0.02, wspace=0.08, left=0.02, right=0.98, top=0.93, bottom=0.05)
    for i, ex in enumerate(examples):
        row = df.iloc[ex["ad_id"] - 1]
        assert row["ground_truth_label"] == ex["true"], f"dataset row does not match ad_id {ex['ad_id']}"
        cell = outer[i // 3, i % 3].subgridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.04)
        ax_img, ax_txt = fig.add_subplot(cell[0]), fig.add_subplot(cell[1])
        ax_img.imshow(Image.open(row["image_path"]).convert("RGB"))
        ax_img.axis("off")
        ax_txt.axis("off")
        correct = ex["true"] == ex["pred"]
        color = GREEN if correct else RED
        ax_txt.text(0, 1.0, clipped(f'"{row["ad_text"]}"', 60, 2), va="top", fontsize=8.5, color=TEXT_PRIMARY,
                    transform=ax_txt.transAxes)
        result = f'{"✓" if correct else "✗"} True: {ex["true"]} /\nPredicted: {ex["pred"]} ({ex["conf"]:.2f})'
        ax_txt.text(0, 0.80, result, va="top", fontsize=9, fontweight="bold", color=color, linespacing=1.3,
                    transform=ax_txt.transAxes)
        if ex["pred"] != "approved" and ex["ad_id"] in explanations:
            ax_txt.text(0, 0.54, clipped(first_sentence(explanations[ex["ad_id"]]), 66, 6), va="top", fontsize=8,
                        color=TEXT_SECONDARY, transform=ax_txt.transAxes)
        else:
            ax_txt.text(0, 0.54, "Not flagged, so no explanation is generated.", va="top", fontsize=8,
                        color=TEXT_SECONDARY, style="italic", transform=ax_txt.transAxes)
    fig.suptitle("Example predictions on test ads (joint CLIP classifier)", fontsize=12, color=TEXT_PRIMARY, y=0.985)
    fig.text(0.5, 0.012, "Photos: Oxford-IIIT Pet Dataset, CC BY-SA 4.0", ha="center", fontsize=9, color=TEXT_SECONDARY)
    plt.savefig("docs/images/example_predictions.png", dpi=130, facecolor="white")
    plt.close()
    print("wrote docs/images/example_predictions.png for ad_ids", [e["ad_id"] for e in examples])


if __name__ == "__main__":
    make_confusion_matrix()
    make_ablation_comparison()
    make_compression_accuracy()
    make_example_predictions()
