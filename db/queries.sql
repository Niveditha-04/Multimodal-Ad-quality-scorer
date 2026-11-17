-- Reference queries against the ads, scores and policy_categories schema.
--
-- The scores table holds rows from more than one model_version, so every query
-- filters by model_version. Mixing arms gives numbers that belong to neither
-- (a blended "misleading recall" of 39.6%, for example, when the arms' real
-- values are 79.2% and 0%).
--
-- model_version values after `python -m db.populate_eval_scores` and
-- `python -m ab_test.run_comparison`:
--   joint_v2_97565ac7  joint classifier, 96 held-out test predictions
--                              (tag + first 8 hex of sha256(model/classifier_head.pt))
--   rule_baseline_v1_d1041021  rule-based baseline, the same 96 test ads
--                              (tag + first 8 hex of sha256(ab_test/rule_based_baseline.py);
--                              `python -m ab_test.baseline_version` prints the current value)
-- Live /score calls add rows under the joint model's version with ad_id above
-- 480 and no ground truth, so queries that need ground truth use ad_id <= 480.
--
-- The queries run unchanged on SQLite and Postgres. Postgres has no
-- round(double precision, integer), so ROUND(x, n) is applied to a CAST to
-- NUMERIC, which is standard SQL. Tied rows can print in a different order, and
-- SQLite drops trailing zeros (0.962) where Postgres keeps them (0.9620).
--
-- Sample output is from `sqlite3 db/ads.db < db/queries.sql` on a database
-- built from the repository: 96 test predictions per arm and no live traffic.

-- 1. Share of the joint classifier's predictions per label.
SELECT
    predicted_label,
    COUNT(*) AS n_predictions,
    ROUND(100.0 * COUNT(*) / (
        SELECT COUNT(*) FROM scores WHERE model_version = 'joint_v2_97565ac7'
    ), 2) AS pct_of_arm_scores
FROM scores
WHERE model_version = 'joint_v2_97565ac7'
GROUP BY predicted_label
ORDER BY n_predictions DESC;

-- sample output:
-- predicted_label  n_predictions  pct_of_arm_scores
-- approved         27             28.13
-- low_quality      24             25.0
-- misleading       23             23.96
-- policy_violation 22             22.92

-- 2. Confidence by predicted label, joint classifier only. The baseline's
--    "confidence" is a constant 1.0 whenever a rule fires, not a probability,
--    so it is not comparable.
SELECT
    predicted_label,
    ROUND(CAST(AVG(confidence) AS NUMERIC), 4) AS avg_confidence,
    ROUND(CAST(MIN(confidence) AS NUMERIC), 4) AS min_confidence,
    ROUND(CAST(MAX(confidence) AS NUMERIC), 4) AS max_confidence,
    COUNT(*) AS n
FROM scores
WHERE model_version = 'joint_v2_97565ac7'
GROUP BY predicted_label
ORDER BY avg_confidence DESC;

-- sample output:
-- predicted_label  avg_confidence  min_confidence  max_confidence  n
-- low_quality      0.9451          0.6459          0.9975          24
-- policy_violation 0.9346          0.6175          0.9929          22
-- approved         0.7339          0.4231          0.9709          27
-- misleading       0.69            0.4225          0.962           23

-- 3. False positives among approved ads, joint classifier only: of the approved
--    ads it scored, what fraction was flagged, by the label it predicted. The
--    denominator counts approved ads scored by this arm only.
SELECT
    s.predicted_label AS incorrectly_predicted_as,
    COUNT(*) AS n_false_positives,
    ROUND(
        100.0 * COUNT(*) / (
            SELECT COUNT(DISTINCT a2.id)
            FROM ads a2 JOIN scores s2 ON s2.ad_id = a2.id
            WHERE a2.ground_truth_label = 'approved' AND s2.model_version = 'joint_v2_97565ac7'
        ),
        2
    ) AS pct_of_scored_approved_ads
FROM scores s
JOIN ads a ON a.id = s.ad_id
WHERE a.ground_truth_label = 'approved'
  AND s.predicted_label != 'approved'
  AND s.model_version = 'joint_v2_97565ac7'
GROUP BY s.predicted_label
ORDER BY n_false_positives DESC;

-- sample output:
-- incorrectly_predicted_as  n_false_positives  pct_of_scored_approved_ads
-- misleading                3                  12.5

-- 4. Per-category recall, joint classifier only: of the ads whose ground truth
--    is each violation category, what fraction was caught. Matches
--    model/results/joint_results.json (0.792, 0.917, 1.000).
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
  AND s.model_version = 'joint_v2_97565ac7'
  AND s.ad_id <= 480  -- labeled eval set only; live traffic (ad_id > 480) has no ground truth
GROUP BY a.ground_truth_label
ORDER BY recall_pct ASC;

-- sample output:
-- ground_truth_label  n_actual  n_correctly_caught  recall_pct
-- misleading          24        19                  79.17
-- policy_violation    24        22                  91.67
-- low_quality         24        24                  100.0

-- 5. The same recall query for the rule-based baseline. Only the model_version
--    filter differs from query 4. The baseline has no way to compare image
--    content with text claims, so its recall on misleading is 0. Matches
--    ab_test/results/comparison.json.
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
  AND s.model_version = 'rule_baseline_v1_d1041021'
GROUP BY a.ground_truth_label
ORDER BY recall_pct ASC;

-- sample output (low_quality and policy_violation are tied, so their order can vary):
-- ground_truth_label  n_actual  n_correctly_caught  recall_pct
-- misleading          24        0                   0.0
-- low_quality         24        24                  100.0
-- policy_violation    24        24                  100.0
