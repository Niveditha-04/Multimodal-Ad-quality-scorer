# scores table: row populations and model_version convention

The `scores` table accumulates rows from three different sources across
this project. Nothing in the schema stops you from mixing them together.
Any query that does not filter explicitly will silently blend populations
that are not comparable. Read this before writing a new query against
`scores`.

## What is in there right now

| model_version | ad_id range | What it is | Row count |
|---|---|---|---|
| `joint_v2_postfix_<hash>` | 1-480, restricted to the 96 test IDs in `data/split.json` | Held-out eval predictions from the trained joint classifier. Written by `db/populate_eval_scores.py`. | 96 |
| `joint_v2_postfix_<hash>` | 481+ | Live `/score` API calls. Real traffic, arbitrary images and text, no ground truth. | Grows over time |
| `rule_baseline_v1_<hash>` | 1-480, restricted to the same 96 test IDs | Phase 5 rule-based baseline (Arm A) predictions on the identical test set. Written by `ab_test/run_comparison.py`. | 96 |

Both `<hash>` suffixes are the first 8 hex characters of a sha256 hash. The
hash is computed over the file that actually determines that arm's
behavior. For the joint model, that file is `classifier_head.pt`
(`model/model_version.py`). For the rule engine, it is
`ab_test/rule_based_baseline.py` (`ab_test/baseline_version.py`). The
reasoning is the same in both places. If either arm is ever retuned or
retrained, its version string changes automatically. Old and new runs can
never silently collide under the same version string.

Early in Phase 5, the baseline was written under a bare `rule_baseline_v1`
string with no hash. This was caught as an inconsistency with the joint
model's convention during the final audit. It was fixed to match, and the
affected DB rows and `db/queries.sql` were regenerated to match.

## Known limitation: explanations inherit the classifier's error rate

This was tested explicitly before Phase 5. See
`rag/results/misclassification_explanation_test.json` for the full test.
When the classifier's prediction is wrong, the RAG plus LLM explanation
layer does not catch it. `retrieve_context()` guarantees a matching policy
rule for the predicted label when that label is `misleading` or
`low_quality`. This means a misclassified ad retrieves the same
confident-looking grounding context as a genuine violation.

In one real test case, an accurate cat-product ad was wrongly classified
as "misleading" by the model. Claude then produced a fluent, specific,
and entirely fabricated justification for the wrong label. It claimed no
product was visible in the image, rather than expressing any doubt.

The explanation layer explains the model's decision. It does not
independently verify that decision. This is stated as a known limitation
in the top-level README. It is not hidden.

## The rule going forward

Every query against `scores` must filter by `model_version` to select
which arm it is reading. It should usually also filter by `ad_id IN
(test_ad_ids from split.json)`, or explicitly note that it includes live
traffic. Without this filter, a metric like "average confidence" or
"accuracy" will silently blend eval-set predictions with unlabeled live
API calls, or blend two different model arms into one meaningless average.

This is not a hypothetical risk. It actually happened. The original
queries 1 and 2 in `db/queries.sql` did not filter by `model_version`.
This was fine while only one model version existed. The moment Phase 5
added `rule_baseline_v1_<hash>`, those two queries started silently
blending both arms together. For example, "misleading recall" came out to
a meaningless 39.6%. That number is a blend of Arm B's real recall (79.2%)
and Arm A's real recall (0%). All 5 queries in `db/queries.sql` now filter
explicitly by `model_version`. This history is the reason why, not a
hypothetical to guard against.
