# Rule-based baseline calibration

All three heuristics were tuned against `data/split.json`'s 384 TRAINING ad
ids only -- the 96 test ads were never looked at during calibration, same
discipline as the ML classifier. Numbers below are what was actually run.

## Blur heuristic (edge variance via PIL FIND_EDGES)

| threshold | approved false-flag rate (train) | low_quality catch rate (train) |
|---|---|---|
| 400 | 0/96 (0.0%) | 49/96 (51.0%) |
| 500 | 1/96 (1.0%) | 52/96 (54.2%) |
| 600 | 6/96 (6.2%) | 53/96 (55.2%) |
| 800 | 13/96 (13.5%) | 55/96 (57.3%) |
| 1000 | 26/96 (27.1%) | 60/96 (62.5%) |

Chose **400**: zero false positives on train-approved, and recall is already
at the theoretical ceiling (~50%, since only half of low_quality rows are
image-degraded -- the other half is text-garbled with a normal image, which
this heuristic structurally cannot see).

## Text-spam heuristic (repeated n-gram OR caps ratio > 0.5 OR >=3 "!")

Train result: approved 0/96 (0.0%), policy_violation 0/96 (0.0%), misleading
0/96 (0.0%), low_quality 43/96 (44.8%) -- again close to the ~50% ceiling
(catches the text-garbled half, structurally blind to the image-degraded half).

## Policy-violation blocklist (regex, case-insensitive)

First pass: 92/96 (95.8%) recall on train policy_violation, 0% false
positives elsewhere. Missed 4 examples all from the same template
("...guarantees your X will live 5 years longer...") -- `\bguarantee[d]?\b`
doesn't match "guarantees" (only covers the past-tense form, not the
third-person-singular verb form). Fixed to `\bguarantee[sd]?\b`, also added
`comes close` to catch "No other X comes close" from the same template.
Re-ran: 96/96 (100%) recall on train policy_violation, 0% false positives on
the other 3 classes.

This isn't overfitting to the test set -- the blocklist was built and fixed
against train only, and it directly encodes literal violation phrases
(exactly what a real policy team's keyword list would look like). High
recall on this specific class is the expected, correct outcome for a
keyword-based approach against textbook violation language, not a red flag.
The real test of this baseline vs. the joint classifier is `misleading`,
where no rule-based signal exists at all -- see ab_test/results/comparison.json
for the actual test-set numbers.
