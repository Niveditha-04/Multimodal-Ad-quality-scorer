# Golden set core finding: "misleading" is the model's error catch-all

Confirmed by human spot-check against the actual images (all 4 cases below),
then verified against the full Phase 2 confusion matrix, not just these 4
examples in isolation.

## The 4 misclassification cases in this golden set

All 4 of the test set's ads where the classifier's prediction disagrees with
ground truth are represented here (ad_ids 6, 97, 105, 146) -- not a
cherry-picked subset. All 4 were predicted `misleading`:

| ad_id | ground truth | predicted | confidence |
|---|---|---|---|
| 6 | approved | misleading | 0.73 |
| 97 | approved | misleading | 0.70 |
| 105 | approved | misleading | 0.51 |
| 146 | policy_violation | misleading | 0.42 |

## Why this isn't just 4 anecdotes

Cross-checked against the joint classifier's full 96-ad test-set confusion
matrix (`model/results/metrics.json`, `[[21,0,3,0],[1,22,1,0],[5,0,19,0],[0,0,0,24]]`,
labels `[approved, policy_violation, misleading, low_quality]`). Of the 10
total misclassifications in the whole test set:

- **4 are false `misleading` predictions** (3 from truly-`approved` ads, 1
  from a truly-`policy_violation` ad) -- exactly the 4 cases in this golden
  set, with none omitted.
- **6 are false `approved` predictions** (5 from truly-`misleading` ads, 1
  from a truly-`policy_violation` ad) -- the model's other error mode is
  under-flagging, missing real violations entirely.
- **Zero are false `policy_violation` or `low_quality` predictions.** These
  two classes have perfect precision (1.000, confirmed in
  `model/results/joint_results.json`) -- when the model predicts one of
  these two, it is never wrong in this test set.

So the error pattern isn't "the model makes mistakes roughly evenly across
categories" -- it's concentrated specifically in the approved/misleading
boundary. `misleading` functions as a catch-all the model reaches for on
ambiguous or actually-fine ads, not a label it only misapplies on genuine
near-miss species confusions. This matters directly for Phase 8's
hallucination-rate finding: every explanation generated for one of these 4
ads is an explanation for a `misleading` verdict that was wrong in a
specific, patterned way, not a random error -- which is exactly the
scenario the explanation layer's "grounded in predicted_label, not ground
truth" design (see `db/README.md`) fails hardest on.
