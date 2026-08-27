"""Generates docs/images/confusion_matrix.png and docs/images/ablation_comparison.png
from the actual saved results files (model/results/metrics.json,
model/results/ablation_summary.json) -- not illustrative or hand-drawn, the
real numbers already reported in the README and Phase 2 writeup.

Colors follow the dataviz skill's reference palette: the confusion matrix is
a magnitude encoding (sequential single hue, blue, light-to-dark); the
ablation comparison is 3 distinct series (image/text/joint), using the
palette's first 3 categorical slots, which the palette's own validation
notes confirm clear the all-pairs CVD/contrast gates together.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def make_confusion_matrix():
    metrics = json.load(open("model/results/metrics.json"))
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


if __name__ == "__main__":
    make_confusion_matrix()
    make_ablation_comparison()
