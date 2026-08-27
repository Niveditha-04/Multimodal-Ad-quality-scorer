"""Builds eval/golden_set.json: 32 flagged test-set ads with independently
determined ground truth, for scoring generated explanations against.

Selection: every ad here has predicted_label != 'approved' from the joint
classifier (model_version=joint_v2_postfix_97565ac7) on the 96-ad test set --
these are exactly the ads that would actually trigger explanation generation
through /score, since approved ads never get one. Includes ALL 4 known
misclassification cases in this test set (ad_ids 6, 97, 105, 146 -- ground
truth is genuinely compliant/a different violation than what the model
predicted), not just the one from the v1 README, plus a confidence-spread
sample of true positives from each of the three violation categories.

Ground truth for species/breed comes from data/dataset.csv's
source_species/source_breed columns -- recorded directly from the Oxford
Pet filename convention at dataset-construction time, independent of any
model or LLM inference, so it isn't vulnerable to the failure mode where
the same reasoning that generates an explanation also grades it. A sample
of 8 of these 32 (all 4 misclassification cases plus one true positive from
each category, plus one of each low_quality subtype) was additionally
visually confirmed by viewing the actual image files directly -- not just
trusting the metadata -- before this script was finalized.

Which policy rule genuinely applies is determined by matching each ad's
actual text against the known template categories from
data/build_dataset.py (HEALTH_CLAIM_TEMPLATES, URGENCY_CLAIM_TEMPLATES,
SUPERLATIVE_CLAIM_TEMPLATES, and the fear-based "Stop your ... suffering"
template that maps to prohibited_language), cross-checked against the
actual rule text in db/policy_categories.py.
"""
import json

import pandas as pd

# ad_id -> which policy_categories.py category_name genuinely applies,
# independent of what the model predicted. "none" = genuinely compliant ad.
GENUINE_RULE = {
    6: "none",  # approved, wrongly predicted misleading -- the v1 README's known case
    97: "none",  # approved, wrongly predicted misleading
    105: "none",  # approved, wrongly predicted misleading
    146: "unsubstantiated_health_claims",  # "guarantees...5 years longer" -- wrongly predicted misleading
    300: "mismatched_creative", 249: "mismatched_creative", 333: "mismatched_creative",
    321: "mismatched_creative", 347: "mismatched_creative", 277: "mismatched_creative",
    243: "mismatched_creative", 320: "mismatched_creative", 359: "mismatched_creative",
    185: "prohibited_language", 157: "prohibited_language",
    178: "unsubstantiated_health_claims", 168: "unsubstantiated_health_claims", 161: "unsubstantiated_health_claims",
    170: "fake_urgency_or_scarcity", 144: "fake_urgency_or_scarcity", 145: "fake_urgency_or_scarcity",
    140: "absolute_superlative_claims", 211: "absolute_superlative_claims",
    418: "low_quality_creative", 436: "low_quality_creative", 388: "low_quality_creative",
    412: "low_quality_creative", 396: "low_quality_creative", 450: "low_quality_creative",
    461: "low_quality_creative", 429: "low_quality_creative", 367: "low_quality_creative",
}

# subtype only meaningful for low_quality ads: which half of the ad is
# actually degraded (from build_dataset.py's construction -- text-garbled
# rows use LOW_QUALITY_TEXT_TEMPLATES with a normal image; image-degraded
# rows use a normal APPROVED_TEMPLATES-style sentence with a blurred image)
LOW_QUALITY_SUBTYPE = {
    418: "text", 436: "text", 388: "text", 412: "text", 396: "text", 450: "text",
    461: "image", 429: "image", 367: "image",
}

VISUALLY_CONFIRMED = {6, 97, 105, 146, 300, 185, 418, 429}


def main():
    df = pd.read_csv("data/dataset.csv")
    df["ad_id"] = df.index + 1
    ad_ids = list(GENUINE_RULE.keys())
    sub = df[df["ad_id"].isin(ad_ids)].set_index("ad_id").loc[ad_ids]

    # predicted label + confidence, from the actual DB (joint classifier, test set)
    import sqlite3
    conn = sqlite3.connect("db/ads.db")
    cur = conn.cursor()
    placeholders = ",".join(str(i) for i in ad_ids)
    cur.execute(
        f"SELECT ad_id, predicted_label, confidence FROM scores "
        f"WHERE model_version='joint_v2_postfix_97565ac7' AND ad_id IN ({placeholders})"
    )
    preds = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
    conn.close()

    golden_set = []
    for ad_id, row in sub.iterrows():
        genuine_rule = GENUINE_RULE[ad_id]
        pred_label, pred_conf = preds[ad_id]
        entry = {
            "ad_id": int(ad_id),
            "image_path": row["image_path"],
            "ad_text": row["ad_text"],
            "ground_truth_label": row["ground_truth_label"],
            "model_predicted_label": pred_label,
            "model_confidence": pred_conf,
            "is_misclassification": row["ground_truth_label"] != pred_label,
            "ground_truth_species_shown": row["source_species"],
            "ground_truth_breed_shown": row["source_breed"],
            "ground_truth_genuine_policy_rule": genuine_rule,
            "ground_truth_low_quality_subtype": LOW_QUALITY_SUBTYPE.get(ad_id),
            "visually_confirmed": ad_id in VISUALLY_CONFIRMED,
        }
        golden_set.append(entry)

    with open("eval/golden_set.json", "w") as f:
        json.dump(golden_set, f, indent=2)

    print(f"wrote {len(golden_set)} entries to eval/golden_set.json")
    n_misclass = sum(1 for e in golden_set if e["is_misclassification"])
    print(f"  {n_misclass} misclassification cases (model prediction != ground truth)")
    print(f"  {sum(1 for e in golden_set if e['visually_confirmed'])} visually confirmed against the image file")
    from collections import Counter
    print("  by ground_truth_label:", Counter(e["ground_truth_label"] for e in golden_set))
    print("  by model_predicted_label:", Counter(e["model_predicted_label"] for e in golden_set))


if __name__ == "__main__":
    main()
