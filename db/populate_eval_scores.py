"""Writes the joint classifier's predictions on the held-out test set into
the scores table, so db/queries.sql (violation rate, confidence by label,
false-positive rate, per-category recall) can be run against real evaluation
data instead of staying untestable until live API traffic accumulates.

Refuses to run twice for the same model_version, same reasoning as
load_dataset.py -- avoids silently duplicating rows on a re-run.
"""
import json
import sys

import numpy as np
import torch

from db.models import Score
from db.session import SessionLocal
from model.clip_features import build_joint_features
from model.classifier_architecture import LABELS, MLPHead, HIDDEN_DIM, DROPOUT
from model.model_version import get_model_version


def main() -> None:
    model_version = get_model_version()
    session = SessionLocal()

    split = json.loads(open("data/split.json").read())
    test_ids = set(split["test_ad_ids"])

    # idempotency check is scoped to test-set ad ids specifically, not "any
    # score with this model_version" -- live /score API traffic writes rows
    # under the same model_version, and those shouldn't block a re-run of
    # eval-set scoring (or look like it already happened)
    existing = session.query(Score).filter(
        Score.model_version == model_version, Score.ad_id.in_(test_ids),
    ).count()
    if existing > 0:
        print(f"scores already has {existing} eval rows for model_version={model_version} -- skipping")
        session.close()
        sys.exit(0)

    d = np.load("model/embeddings.npz", allow_pickle=True)
    ad_ids = d["ad_ids"]
    features = build_joint_features(d["image_embeds"], d["text_embeds"])

    test_mask = np.array([i in test_ids for i in ad_ids])
    test_ad_ids = ad_ids[test_mask]
    test_features = features[test_mask]
    assert len(test_ad_ids) == len(test_ids), "embeddings.npz doesn't cover every test ad id"

    classifier = MLPHead(input_dim=1024, hidden_dim=HIDDEN_DIM, n_classes=len(LABELS), dropout=DROPOUT)
    classifier.load_state_dict(torch.load("model/classifier_head.pt", map_location="cpu"))
    classifier.eval()

    with torch.no_grad():
        logits = classifier(torch.tensor(test_features, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)

    for ad_id, pred_idx, prob_row in zip(test_ad_ids, preds, probs):
        session.add(Score(
            ad_id=int(ad_id),
            predicted_label=LABELS[pred_idx],
            confidence=float(prob_row[pred_idx]),
            model_version=model_version,
        ))
    session.commit()
    print(f"wrote {len(test_ad_ids)} eval scores for model_version={model_version}")
    session.close()


if __name__ == "__main__":
    main()
