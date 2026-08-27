# Rule-based baseline calibration

All three heuristics were tuned only on the 384 training ads listed in
`data/split.json`. The 96 test ads were never looked at during calibration.
This matches the same rule used for the ML classifier. The numbers below
are from real runs, not estimates.

## Blur heuristic

This heuristic measures image sharpness using PIL's FIND_EDGES filter. A
sharper image has more edge variance. A blurrier image has less.

| Threshold | Approved false-flag rate (train) | Low-quality catch rate (train) |
|---|---|---|
| 400 | 0/96 (0.0%) | 49/96 (51.0%) |
| 500 | 1/96 (1.0%) | 52/96 (54.2%) |
| 600 | 6/96 (6.2%) | 53/96 (55.2%) |
| 800 | 13/96 (13.5%) | 55/96 (57.3%) |
| 1000 | 26/96 (27.1%) | 60/96 (62.5%) |

We chose 400. It produces zero false positives on approved training ads.
Its recall is already close to the highest possible rate for this
heuristic. Only about half of `low_quality` rows are image-degraded. The
other half have garbled text but a normal image. This heuristic cannot see
that half, so it cannot catch it no matter the threshold.

## Text-spam heuristic

This heuristic flags text that repeats an n-gram (a short sequence of
words), has a capital-letter ratio above 0.5, or contains 3 or more
exclamation points.

Train result: approved 0/96 (0.0%), policy_violation 0/96 (0.0%),
misleading 0/96 (0.0%), low_quality 43/96 (44.8%). This is close to the
same ~50% ceiling as the blur heuristic. It catches the text-garbled half
of `low_quality` and is structurally blind to the image-degraded half.

## Policy-violation blocklist

This is a case-insensitive regex blocklist of banned phrases.

First pass: 92/96 (95.8%) recall on train `policy_violation`, 0% false
positives elsewhere. It missed 4 examples, all from the same template:
"...guarantees your X will live 5 years longer...". The regex
`\bguarantee[d]?\b` does not match the word "guarantees". It only matches
the past-tense form "guaranteed", not the third-person-singular verb form
"guarantees". We fixed this to `\bguarantee[sd]?\b`. We also added the
phrase `comes close` to catch "No other X comes close" from the same
template.

After the fix: 96/96 (100%) recall on train `policy_violation`, 0% false
positives on the other 3 classes.

This is not overfitting to the test set. The blocklist was built and fixed
using only the training data. It directly encodes literal violation
phrases, which is exactly what a real policy team's keyword list would
look like. High recall on this one class is the expected, correct outcome
for a keyword-based approach against textbook violation language. It is
not a red flag.

The real test of this baseline against the joint classifier is the
`misleading` category. No rule-based signal exists for that category at
all. See `ab_test/results/comparison.json` for the actual test-set numbers.
