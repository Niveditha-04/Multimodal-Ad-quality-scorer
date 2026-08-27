# scores table: row populations and model_version convention

`scores` is going to accumulate rows from three different sources across
this project. Nothing in the schema stops you from aggregating across all of
them, so any query that doesn't filter explicitly is silently mixing
populations that aren't comparable. Read this before writing a new query
against `scores`.

## What's in there right now

| model_version | ad_id range | what it is | row count |
|---|---|---|---|
| `joint_v2_postfix_<hash>` | 1-480, restricted to the 96 `data/split.json` test IDs | held-out eval predictions from the trained joint classifier, written by `db/populate_eval_scores.py` | 96 |
| `joint_v2_postfix_<hash>` | 481+ | live `/score` API calls (real traffic, arbitrary images/text, no ground truth) | grows over time |
| `rule_baseline_v1_<hash>` | 1-480, restricted to the same 96 test IDs | Phase 5 rule-based baseline (Arm A) predictions on the identical test set, written by `ab_test/run_comparison.py` | 96 |

Both `<hash>` suffixes are the first 8 hex chars of a sha256 over the file
that actually determines the arm's behavior -- `classifier_head.pt` for the
joint model (`model/model_version.py`), `ab_test/rule_based_baseline.py` for
the rule engine (`ab_test/baseline_version.py`). Same reasoning both places:
if either arm is ever retuned/retrained, the version string changes
automatically, so old and new runs can't silently collide. Early in Phase 5
the baseline was written under a bare `rule_baseline_v1` string with no
hash -- caught as an inconsistency with the joint model's convention during
the final audit, fixed to match, and the DB rows + `db/queries.sql` were
regenerated/updated accordingly.

## Known limitation: explanations inherit the classifier's error rate

Tested explicitly before Phase 5 (see `rag/results/misclassification_explanation_test.json`):
when the classifier's prediction is wrong, the RAG+LLM explanation layer does
NOT catch it. `retrieve_context()` guarantees the policy rule matching
`predicted_label` for `misleading`/`low_quality`, so a misclassified ad
retrieves the same confident-looking grounding context as a genuine
violation. On a real test case (an accurate cat-product ad the classifier
wrongly called "misleading"), Claude produced a fluent, specific, and
entirely fabricated justification (claimed no product was visible in the
image) rather than any hedge. The explanation layer explains the model's
decision -- it does not independently verify it. This is stated as a known
limitation in the top-level README, not silently absorbed.

## The rule, going forward

Every query against `scores` must filter by **`model_version`** (which arm),
and should usually also filter by **`ad_id IN (test_ad_ids from
split.json)`** or explicitly note that it's including live traffic --
otherwise "average confidence" or "accuracy" silently blends eval-set
predictions with unlabeled live API calls, or blends two different model
arms into one meaningless average.

This actually happened: `db/queries.sql`'s original queries 1-2 didn't
filter by model_version, which was fine while only one version existed, but
started silently blending both arms the moment Phase 5 landed
`rule_baseline_v1_<hash>` -- e.g. "misleading recall" came out to a
meaningless 39.6% blend of Arm B's real 79.2% and Arm A's real 0%. All 5
queries in `db/queries.sql` now filter explicitly by model_version; this is
the reason why, not a hypothetical to watch out for.
