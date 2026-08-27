"""Simulated offline A/B comparison: rule-based baseline (Arm A) vs. the
trained joint classifier (Arm B), both scored on the identical 96 held-out
test ads. This is NOT a live production A/B test with real traffic -- it's
an offline comparison of two scoring functions against a fixed labeled set.

Arm B's predictions are NOT recomputed here -- they already exist in
`scores` from db/populate_eval_scores.py (Phase 3), so this triggers zero
new model inference and zero LLM calls. Arm A's predictions are computed
fresh (the rule engine is cheap) and written to `scores` under
model_version="rule_baseline_v1" for consistency with the DB convention in
db/README.md.

Metrics reported two ways:
  - per-class precision/recall/F1 + confusion matrix (same shape as Phase 2)
  - binary flagged-vs-approved precision/recall/FPR/FNR (collapsing the 3
    violation categories into "flagged") -- this is the natural framing for
    "how often do we wrongly hold a clean ad" / "how often do we miss a bad
    one", and what the spec's FPR/FNR request actually maps onto for a
    4-class problem

Statistical test is McNemar's, not a plain independent-samples chi-square.
Both arms score the SAME 96 ads -- that's paired data, not two independent
samples, and a naive chi-square-of-independence on marginal correct/incorrect
counts would misstate significance by ignoring the pairing. McNemar's test
(itself a chi-square-distributed statistic, just the paired variant) is the
statistically correct tool here. A bootstrap CI on the accuracy difference is
also reported as a second, independent line of evidence.
"""
import json
import sqlite3

import numpy as np
from PIL import Image
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

from ab_test.baseline_version import get_baseline_version
from ab_test.rule_based_baseline import predict as rule_based_predict
from db.models import Score
from db.session import SessionLocal
from model.classifier_architecture import LABELS
from model.model_version import get_model_version


def load_test_ads():
    split = json.loads(open("data/split.json").read())
    test_ids = set(split["test_ad_ids"])

    conn = sqlite3.connect("db/ads.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT id, image_path, ad_text, ground_truth_label FROM ads WHERE id <= 480")
    rows = [dict(r) for r in cur.fetchall() if r["id"] in test_ids]
    conn.close()
    assert len(rows) == len(test_ids), "mismatch between split.json test ids and ads table"
    return rows


def run_arm_a(test_ads: list[dict]) -> dict[int, str]:
    predictions = {}
    for ad in test_ads:
        image = Image.open(ad["image_path"]).convert("RGB")
        predictions[ad["id"]] = rule_based_predict(ad["ad_text"], image)
    return predictions


def load_arm_b(test_ads: list[dict]) -> dict[int, str]:
    joint_version = get_model_version()
    test_ids = [ad["id"] for ad in test_ads]
    session = SessionLocal()
    rows = session.query(Score).filter(
        Score.model_version == joint_version, Score.ad_id.in_(test_ids),
    ).all()
    session.close()
    predictions = {r.ad_id: r.predicted_label for r in rows}
    assert len(predictions) == len(test_ids), (
        f"expected {len(test_ids)} joint-model scores, found {len(predictions)} "
        f"-- did db/populate_eval_scores.py run for the current model version?"
    )
    return predictions


def persist_arm_a(predictions: dict[int, str]) -> None:
    baseline_version = get_baseline_version()
    session = SessionLocal()
    existing = session.query(Score).filter(Score.model_version == baseline_version).count()
    if existing > 0:
        print(f"scores already has {existing} rows for {baseline_version} -- skipping write")
        session.close()
        return
    for ad_id, label in predictions.items():
        # rule engine is a deterministic boolean trigger, not a calibrated
        # probability model -- confidence=1.0 records "rule fired", it is
        # NOT comparable to the joint model's softmax confidence
        session.add(Score(ad_id=ad_id, predicted_label=label, confidence=1.0, model_version=baseline_version))
    session.commit()
    session.close()
    print(f"wrote {len(predictions)} Arm A scores under model_version={baseline_version}")


def per_class_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "per_class": {
            LABELS[i]: {
                "precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "support": int(support[i]),
            } for i in range(len(LABELS))
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": LABELS,
    }


def binary_flagged_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    # "flagged" = anything other than approved
    true_flagged = np.array([t != "approved" for t in y_true])
    pred_flagged = np.array([p != "approved" for p in y_pred])

    tp = int((true_flagged & pred_flagged).sum())
    fp = int((~true_flagged & pred_flagged).sum())
    fn = int((true_flagged & ~pred_flagged).sum())
    tn = int((~true_flagged & ~pred_flagged).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "precision": precision, "recall": recall,
        "false_positive_rate": fpr, "false_negative_rate": fnr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def mcnemar_test(y_true: list[str], pred_a: list[str], pred_b: list[str]) -> dict:
    a_correct = np.array([a == t for a, t in zip(pred_a, y_true)])
    b_correct = np.array([b == t for b, t in zip(pred_b, y_true)])

    # b = A right, B wrong; c = A wrong, B right -- the two discordant cells
    b_cell = int((a_correct & ~b_correct).sum())
    c_cell = int((~a_correct & b_correct).sum())

    # continuity-corrected McNemar statistic, chi-square(1) distributed
    if b_cell + c_cell == 0:
        return {"b_a_right_b_wrong": b_cell, "c_a_wrong_b_right": c_cell, "statistic": 0.0, "p_value": 1.0}
    statistic = (abs(b_cell - c_cell) - 1) ** 2 / (b_cell + c_cell)
    p_value = float(stats.chi2.sf(statistic, df=1))
    return {"b_a_right_b_wrong": b_cell, "c_a_wrong_b_right": c_cell, "statistic": float(statistic), "p_value": p_value}


def bootstrap_accuracy_diff(y_true: list[str], pred_a: list[str], pred_b: list[str], n_boot: int = 10000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    y_true_arr, pred_a_arr, pred_b_arr = np.array(y_true), np.array(pred_a), np.array(pred_b)
    n = len(y_true_arr)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        acc_a = (pred_a_arr[idx] == y_true_arr[idx]).mean()
        acc_b = (pred_b_arr[idx] == y_true_arr[idx]).mean()
        diffs[i] = acc_b - acc_a
    return {
        "mean_diff_b_minus_a": float(diffs.mean()),
        "ci_95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "excludes_zero": bool(np.percentile(diffs, 2.5) > 0 or np.percentile(diffs, 97.5) < 0),
    }


def main() -> None:
    test_ads = load_test_ads()
    print(f"loaded {len(test_ads)} test ads")

    arm_a_preds = run_arm_a(test_ads)
    print(f"Arm A (rule-based) predictions computed for {len(arm_a_preds)} ads")
    persist_arm_a(arm_a_preds)

    arm_b_preds = load_arm_b(test_ads)
    print(f"Arm B (joint classifier) predictions loaded for {len(arm_b_preds)} ads (reused, not recomputed)")

    y_true = [ad["ground_truth_label"] for ad in test_ads]
    pred_a = [arm_a_preds[ad["id"]] for ad in test_ads]
    pred_b = [arm_b_preds[ad["id"]] for ad in test_ads]

    results = {
        "n_test": len(test_ads),
        "arm_a_rule_based": {
            "per_class": per_class_metrics(y_true, pred_a),
            "binary_flagged_vs_approved": binary_flagged_metrics(y_true, pred_a),
        },
        "arm_b_joint_classifier": {
            "per_class": per_class_metrics(y_true, pred_b),
            "binary_flagged_vs_approved": binary_flagged_metrics(y_true, pred_b),
        },
        "mcnemar_test": mcnemar_test(y_true, pred_a, pred_b),
        "bootstrap_accuracy_diff": bootstrap_accuracy_diff(y_true, pred_a, pred_b),
    }

    print("\n=== Arm A (rule-based) ===")
    print(f"accuracy: {results['arm_a_rule_based']['per_class']['accuracy']:.4f}")
    print(f"binary flagged: {results['arm_a_rule_based']['binary_flagged_vs_approved']}")

    print("\n=== Arm B (joint classifier) ===")
    print(f"accuracy: {results['arm_b_joint_classifier']['per_class']['accuracy']:.4f}")
    print(f"binary flagged: {results['arm_b_joint_classifier']['binary_flagged_vs_approved']}")

    print("\n=== McNemar's test (paired) ===")
    print(results["mcnemar_test"])

    print("\n=== Bootstrap accuracy diff (B - A) ===")
    print(results["bootstrap_accuracy_diff"])

    import os
    os.makedirs("ab_test/results", exist_ok=True)
    with open("ab_test/results/comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved to ab_test/results/comparison.json")


if __name__ == "__main__":
    main()
