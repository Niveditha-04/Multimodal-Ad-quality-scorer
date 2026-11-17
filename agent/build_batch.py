"""Builds agent/batch.json: 18 ads for the campaign auditor to work through.

Most ads come from eval/golden_set.json instead of a fresh random slice of the
test set, because every golden-set ad has independently determined ground
truth (see eval/build_golden_set.py), so each claim the agent makes about an ad
in its summary can be checked against it.

The golden set contains only flagged ads (those that trigger an explanation),
so 4 correctly classified `approved` ads outside the golden set are added to
include "no violation" outcomes.

Composition (18 total):
  4  correctly-classified approved ads (new)
  4  all of this test set's misclassification cases (from the golden set:
     ad_ids 6, 97, 105, 146)
  4  policy_violation true positives (from the golden set)
  3  misleading true positives (from the golden set)
  3  low_quality true positives (from the golden set)
"""
import json

import pandas as pd

FRESH_APPROVED_IDS = [8, 16, 25, 34]
GOLDEN_SET_IDS = [
    6, 97, 105, 146,             # all 4 misclassification cases
    140, 161, 178, 211,          # policy_violation true positives
    243, 300, 359,               # misleading true positives
    367, 388, 429,               # low_quality true positives
]


def main() -> None:
    golden_set = {e["ad_id"]: e for e in json.loads(open("eval/golden_set.json").read())}
    df = pd.read_csv("data/dataset.csv")
    df["ad_id"] = df.index + 1

    batch = []
    for ad_id in FRESH_APPROVED_IDS:
        row = df[df["ad_id"] == ad_id].iloc[0]
        batch.append({
            "ad_id": int(ad_id),
            "image_path": row["image_path"],
            "ad_text": row["ad_text"],
            "ground_truth_label": row["ground_truth_label"],
            "model_predicted_label": "approved",  # confirmed via DB query when selecting these
            "is_misclassification": False,
            "ground_truth_species_shown": row["source_species"],
            "ground_truth_breed_shown": row["source_breed"],
            "ground_truth_genuine_policy_rule": "none",
        })

    for ad_id in GOLDEN_SET_IDS:
        e = golden_set[ad_id]
        batch.append({
            "ad_id": e["ad_id"],
            "image_path": e["image_path"],
            "ad_text": e["ad_text"],
            "ground_truth_label": e["ground_truth_label"],
            "model_predicted_label": e["model_predicted_label"],
            "is_misclassification": e["is_misclassification"],
            "ground_truth_species_shown": e["ground_truth_species_shown"],
            "ground_truth_breed_shown": e["ground_truth_breed_shown"],
            "ground_truth_genuine_policy_rule": e["ground_truth_genuine_policy_rule"],
        })

    batch.sort(key=lambda x: x["ad_id"])

    with open("agent/batch.json", "w") as f:
        json.dump(batch, f, indent=2)

    from collections import Counter
    print(f"wrote {len(batch)} ads to agent/batch.json")
    print("by ground_truth_label:", Counter(b["ground_truth_label"] for b in batch))
    print("by model_predicted_label:", Counter(b["model_predicted_label"] for b in batch))
    print("misclassifications:", sum(1 for b in batch if b["is_misclassification"]))


if __name__ == "__main__":
    main()
