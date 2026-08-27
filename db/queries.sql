-- Real queries against the ads/scores/policy_categories schema.
-- scores currently holds 98 rows: 96 held-out test-set predictions from the
-- joint classifier (model_version=joint_v2_postfix_97565ac7, written by
-- db/populate_eval_scores.py) plus 2 rows from live /score API smoke-test
-- calls made during Phase 3 testing. Actual output from each query is
-- pasted below it -- run against db/ads.db as of 2026-08-27, not simulated.

-- 1. Violation rate by predicted category: of everything scored, what share
--    of predictions fall into each label? Answers "how often is the model
--    flagging things, and for what reason."
SELECT
    predicted_label,
    COUNT(*) AS n_predictions,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM scores), 2) AS pct_of_all_scores
FROM scores
GROUP BY predicted_label
ORDER BY n_predictions DESC;

-- actual output:
-- predicted_label    n_predictions  pct_of_all_scores
-- approved            28            28.57
-- low_quality         24            24.49
-- policy_violation    23            23.47
-- misleading          23            23.47

-- 2. Average confidence by predicted label: sanity check on calibration --
--    if "approved" predictions have lower average confidence than
--    "policy_violation" predictions, that's worth knowing before trusting
--    the model's confidence score for anything downstream (e.g. auto-approve
--    thresholds).
SELECT
    predicted_label,
    ROUND(AVG(confidence), 4) AS avg_confidence,
    ROUND(MIN(confidence), 4) AS min_confidence,
    ROUND(MAX(confidence), 4) AS max_confidence,
    COUNT(*) AS n
FROM scores
GROUP BY predicted_label
ORDER BY avg_confidence DESC;

-- actual output:
-- predicted_label    avg_confidence  min_confidence  max_confidence  n
-- low_quality         0.9451          0.6459          0.9975          24
-- policy_violation    0.9326          0.6175          0.9929          23
-- approved            0.7344          0.4231          0.9709          28
-- misleading          0.6900          0.4225          0.9620          23
-- note: approved and misleading have the lowest average confidence -- these
-- are also the two classes the joint model confuses with each other most
-- (see model/results/joint_results.json confusion matrix), so lower
-- confidence there is consistent with genuine model uncertainty, not noise.

-- 3. False positive rate by ground-truth category: of ads that were actually
--    approved (ground_truth_label = 'approved'), what fraction did the model
--    incorrectly flag as some kind of violation, broken out by what it
--    incorrectly called them? This is the "how often do we wrongly block a
--    clean ad, and as what" query -- the number that matters most for a
--    policy team worried about over-flagging.
-- Denominator is scoped to approved ads that were actually SCORED, not every
-- approved ad in the `ads` table -- most approved ads are training-set rows
-- that never got a score row, so counting all of them would understate the
-- false-positive rate (division by ads that were never eligible to be a
-- false positive in the first place).
SELECT
    s.predicted_label AS incorrectly_predicted_as,
    COUNT(*) AS n_false_positives,
    ROUND(
        100.0 * COUNT(*) / (
            SELECT COUNT(DISTINCT a2.id)
            FROM ads a2 JOIN scores s2 ON s2.ad_id = a2.id
            WHERE a2.ground_truth_label = 'approved'
        ),
        2
    ) AS pct_of_scored_approved_ads
FROM scores s
JOIN ads a ON a.id = s.ad_id
WHERE a.ground_truth_label = 'approved'
  AND s.predicted_label != 'approved'
GROUP BY s.predicted_label
ORDER BY n_false_positives DESC;

-- actual output:
-- incorrectly_predicted_as  n_false_positives  pct_of_scored_approved_ads
-- misleading                 3                  12.5
-- (matches the joint model's confusion matrix exactly: 3 of the 24 scored
-- approved ads were misclassified as misleading, 0 as policy_violation or
-- low_quality)

-- 4. (bonus) Per-category recall: of ads whose ground truth is a given
--    violation category, what fraction did the model correctly catch?
--    Complements query 3 -- that one measures over-flagging, this one
--    measures under-flagging (misses).
SELECT
    a.ground_truth_label,
    COUNT(*) AS n_actual,
    SUM(CASE WHEN s.predicted_label = a.ground_truth_label THEN 1 ELSE 0 END) AS n_correctly_caught,
    ROUND(
        100.0 * SUM(CASE WHEN s.predicted_label = a.ground_truth_label THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS recall_pct
FROM ads a
JOIN scores s ON s.ad_id = a.id
WHERE a.ground_truth_label != 'approved'
GROUP BY a.ground_truth_label
ORDER BY recall_pct ASC;

-- actual output:
-- ground_truth_label  n_actual  n_correctly_caught  recall_pct
-- misleading            24        19                  79.17
-- policy_violation      24        22                  91.67
-- low_quality           24        24                 100.00
-- matches model/results/joint_results.json per-class recall exactly
-- (0.792, 0.917, 1.000) -- this is the same number computed two different
-- ways (sklearn during training, raw SQL against persisted scores here),
-- which is a real cross-check, not a restatement.
