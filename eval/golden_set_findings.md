# Golden set core finding: "misleading" is the model's error catch-all

This finding was confirmed by a human spot-check against the actual images
for all 4 cases below. It was then verified against the full Phase 2
confusion matrix, not just these 4 examples in isolation.

## The 4 misclassification cases in this golden set

The test set has exactly 4 ads where the classifier's prediction disagrees
with ground truth. All 4 are represented here (ad IDs 6, 97, 105, and
146). This is not a cherry-picked subset. All 4 were predicted as
`misleading`.

| Ad ID | Ground truth | Predicted | Confidence |
|---|---|---|---|
| 6 | approved | misleading | 0.73 |
| 97 | approved | misleading | 0.70 |
| 105 | approved | misleading | 0.51 |
| 146 | policy_violation | misleading | 0.42 |

## Why this is not just 4 anecdotes

We cross-checked this against the joint classifier's full 96-ad test-set
confusion matrix, saved in `model/results/metrics.json`. The matrix is
`[[21,0,3,0],[1,22,1,0],[5,0,19,0],[0,0,0,24]]`, with row and column
labels `[approved, policy_violation, misleading, low_quality]`. There are
10 total misclassifications in the whole test set. Here is how they break
down.

| Error type | Count | Detail |
|---|---|---|
| False `misleading` predictions | 4 | 3 from truly-`approved` ads, 1 from a truly-`policy_violation` ad. These are exactly the 4 cases in this golden set. None are omitted. |
| False `approved` predictions | 6 | 5 from truly-`misleading` ads, 1 from a truly-`policy_violation` ad. This is the model's other error mode: under-flagging, meaning it misses real violations entirely. |
| False `policy_violation` or `low_quality` predictions | 0 | These two classes have perfect precision, 1.000, confirmed in `model/results/joint_results.json`. When the model predicts one of these two labels, it is never wrong in this test set. |

The error pattern is not "the model makes mistakes roughly evenly across
categories." It is concentrated specifically at the boundary between
`approved` and `misleading`. The label `misleading` functions as a
catch-all the model reaches for on ambiguous or actually-fine ads. It is
not a label the model only misapplies on genuine near-miss species
confusions.

This matters directly for Phase 8's hallucination-rate finding. Every
explanation generated for one of these 4 ads is an explanation for a
`misleading` verdict that was wrong in a specific, patterned way. It is
not a random error. This is exactly the scenario where the explanation
layer's design fails hardest. That design grounds each explanation in
`predicted_label`, not in ground truth. See `db/README.md` for the full
description of that design choice.
