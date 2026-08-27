-- Real queries against the ads/scores/policy_categories schema.
-- scores now holds 202 rows across TWO model_versions -- see db/README.md
-- for the full convention. As of this file, that's:
--   joint_v2_postfix_97565ac7  (106 rows: 96 held-out eval predictions from
--     db/populate_eval_scores.py, plus 10 live /score and MCP-tool calls made
--     during Phase 3/4/5/6 testing)
--   rule_baseline_v1_7ba4a019  (96 rows: the Phase 5 rule-based baseline's
--     predictions on the same 96 test ads, from ab_test/run_comparison.py --
--     the hash suffix is sha256(ab_test/rule_based_baseline.py)[:8], see
--     ab_test/baseline_version.py, mirroring model/model_version.py's
--     approach so a future retune of the thresholds gets its own version
--     rather than colliding silently with this one)
--
-- Every query below filters by model_version explicitly. An earlier version
-- of this file did NOT, and worked fine when only one model_version
-- existed -- but the moment rule_baseline_v1 was added, the unfiltered
-- queries started silently blending both arms into meaningless composite
-- numbers (e.g. "misleading recall" came out to 39.6%, which is neither
-- arm's real number -- Arm B is 79.2%, Arm A is a hard 0%). Caught this by
-- literally re-running the old queries after Phase 5 landed and comparing
-- against ab_test/results/comparison.json, per db/README.md's own note to
-- re-verify rather than assume. Fixed by adding the filter everywhere below.
--
-- Verified to run identically on both SQLite (db/ads.db) and Postgres
-- (Phase 7 of the v2 extension, db/migrate_to_postgres.py) -- output is
-- numerically identical on both engines; the only observed differences are
-- two cosmetic formatting artifacts, not data or logic discrepancies: (1)
-- tied rows in queries with no secondary ORDER BY key can come out in either
-- order depending on the engine's internal storage order, and (2) SQLite's
-- ROUND() returns a float that drops trailing zeros in text output (0.962),
-- while Postgres's ROUND(CAST(... AS NUMERIC)) preserves fixed decimal
-- places (0.9620) -- same numeric value both ways.
--
-- Actual output below is from db/ads.db (SQLite) as of 2026-08-27, not
-- simulated -- values are identical on Postgres modulo the two cosmetic
-- differences just described.

-- 1. Violation rate by predicted category, for the joint classifier
--    specifically: of everything it scored, what share of predictions fall
--    into each label?
SELECT
    predicted_label,
    COUNT(*) AS n_predictions,
    ROUND(100.0 * COUNT(*) / (
        SELECT COUNT(*) FROM scores WHERE model_version = 'joint_v2_postfix_97565ac7'
    ), 2) AS pct_of_arm_scores
FROM scores
WHERE model_version = 'joint_v2_postfix_97565ac7'
GROUP BY predicted_label
ORDER BY n_predictions DESC;

-- actual output:
-- predicted_label    n_predictions  pct_of_arm_scores
-- approved            29            27.36
-- misleading          27            25.47
-- policy_violation    25            23.58
-- low_quality         25            23.58
-- (106 total rows for this arm: 96 eval + 10 live/MCP test calls, spread
-- roughly as expected across categories; policy_violation/low_quality are
-- tied at 25 -- see the tie-break note above)

-- 2. Average confidence by predicted label, joint classifier only --
--    sanity check on calibration. Mixing in rule_baseline_v1 here would be
--    especially misleading since that arm's "confidence" isn't a real
--    probability at all (see db/README.md / ab_test/rule_based_baseline.py)
--    -- it's a hardcoded 1.0 for every rule-fired prediction.
-- ROUND()'s 2-argument form needs an explicit CAST to NUMERIC for Postgres
-- (Postgres has no round(double precision, integer) overload -- SQLite's
-- loose typing accepts it directly, Postgres doesn't). CAST(... AS NUMERIC)
-- is standard SQL, not Postgres-specific ::numeric shorthand, so it works
-- identically on both engines -- found this the hard way, running this
-- query against Postgres for the first time errored with "function
-- round(double precision, integer) does not exist" until this cast was added.
SELECT
    predicted_label,
    ROUND(CAST(AVG(confidence) AS NUMERIC), 4) AS avg_confidence,
    ROUND(CAST(MIN(confidence) AS NUMERIC), 4) AS min_confidence,
    ROUND(CAST(MAX(confidence) AS NUMERIC), 4) AS max_confidence,
    COUNT(*) AS n
FROM scores
WHERE model_version = 'joint_v2_postfix_97565ac7'
GROUP BY predicted_label
ORDER BY avg_confidence DESC;

-- actual output:
-- predicted_label    avg_confidence  min_confidence  max_confidence  n
-- low_quality         0.9464          0.6459          0.9975          25
-- policy_violation    0.9292          0.6175          0.9929          25
-- approved            0.7266          0.4231          0.9709          29
-- misleading          0.6822          0.4225          0.9620          27
-- same pattern as before Phase 5: approved and misleading are the two
-- lowest-confidence, lowest-precision-against-each-other classes (see
-- model/results/joint_results.json confusion matrix) -- confidence is
-- tracking genuine model uncertainty, not noise.

-- 3. False positive rate by ground-truth category, joint classifier only:
--    of ads that were actually approved, what fraction did the model
--    incorrectly flag, broken out by what it incorrectly called them?
--    Denominator scoped to approved ads that were actually scored BY THIS
--    ARM specifically, not all approved ads in the table.
SELECT
    s.predicted_label AS incorrectly_predicted_as,
    COUNT(*) AS n_false_positives,
    ROUND(
        100.0 * COUNT(*) / (
            SELECT COUNT(DISTINCT a2.id)
            FROM ads a2 JOIN scores s2 ON s2.ad_id = a2.id
            WHERE a2.ground_truth_label = 'approved' AND s2.model_version = 'joint_v2_postfix_97565ac7'
        ),
        2
    ) AS pct_of_scored_approved_ads
FROM scores s
JOIN ads a ON a.id = s.ad_id
WHERE a.ground_truth_label = 'approved'
  AND s.predicted_label != 'approved'
  AND s.model_version = 'joint_v2_postfix_97565ac7'
GROUP BY s.predicted_label
ORDER BY n_false_positives DESC;

-- actual output:
-- incorrectly_predicted_as  n_false_positives  pct_of_scored_approved_ads
-- misleading                 3                  12.5
-- (unchanged from before Phase 5 -- the 9 additional live-traffic rows
-- under this model_version weren't approved-ground-truth ads, so this
-- query's denominator/numerator are untouched by them)

-- 4. Per-category recall, joint classifier only: of ads whose ground truth
--    is a given violation category, what fraction did the model catch?
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
  AND s.model_version = 'joint_v2_postfix_97565ac7'
  AND s.ad_id <= 480  -- restrict to the labeled eval set -- live traffic (ad_id over 480) has no ground truth to check against
GROUP BY a.ground_truth_label
ORDER BY recall_pct ASC;

-- actual output:
-- ground_truth_label  n_actual  n_correctly_caught  recall_pct
-- misleading            24        19                  79.17
-- policy_violation      24        22                  91.67
-- low_quality           24        24                 100.00
-- unchanged from before Phase 5, and still matches
-- model/results/joint_results.json exactly (0.792, 0.917, 1.000) -- same
-- number, two independent computations (sklearn at training time, raw SQL
-- against persisted scores here).

-- 5. (Phase 5) Same per-category recall query, but for the rule-based
--    baseline (Arm A) instead -- the query is identical except for the
--    model_version filter. Put side by side with query 4, this is the
--    single clearest illustration of what the two arms actually differ on.
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
  AND s.model_version = 'rule_baseline_v1_7ba4a019'
GROUP BY a.ground_truth_label
ORDER BY recall_pct ASC;

-- actual output:
-- ground_truth_label  n_actual  n_correctly_caught  recall_pct
-- misleading            24        0                    0.00
-- low_quality           24        24                 100.00
-- policy_violation      24        24                 100.00
-- (low_quality/policy_violation are tied at 100% -- their relative order is
-- an unstable tie-break, not a meaningful ranking)
-- the rule baseline is perfect on the two categories it has a real signal
-- for (keyword language, image/text degradation) and a hard structural
-- zero on misleading -- it has no mechanism to compare image content
-- against text claims. Matches ab_test/results/comparison.json exactly.
