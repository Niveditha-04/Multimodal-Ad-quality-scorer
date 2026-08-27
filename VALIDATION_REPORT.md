# Validation Report

This document consolidates every verification effort across this project
into one place: the ablation study, the LLM evaluation harness, the
guardrails testing, and the agent reproducibility check. Every number
here comes from a checked-in results file. None are estimated. Detailed
writeups for each area live in their own README files, linked below each
section.

## 1. Ablation study: does the model actually use both modalities

We trained three classifier-head variants on the identical 384-to-96
train and test split: image-only, text-only, and joint, meaning the
concatenation of both modalities. All three used frozen CLIP embeddings.
None fine-tuned the CLIP backbone itself.

| Modality | Accuracy | 95% confidence interval | approved | policy_violation | misleading | low_quality |
|---|---|---|---|---|---|---|
| Image-only | 36.5% | 27.5% to 46.4% | 0.208 | 0.458 | 0.292 | 0.500 |
| Text-only | 68.8% | 58.9% to 77.1% | 0.542 | 1.000 | 0.625 | 0.583 |
| Joint | 89.6% | 81.9% to 94.2% | 0.875 | 0.917 | 0.792 | 1.000 |

The three overall-accuracy confidence intervals do not overlap, so the
ranking of joint over text over image is statistically solid. Two
findings stand out at the per-class level. The `misleading` category
needs both modalities: the joint model's recall clearly beats both single
modalities, since detecting a species mismatch is inherently relational.
The `policy_violation` category does not need the image: text-only recall
(1.000) actually edges out the joint model (0.917), reported as-is rather
than smoothed over.

Full detail: `README.md`, section "Ablation: does the model actually use
both modalities?", and `model/results/ablation_summary.json`.

## 2. LLM evaluation harness: hallucination rate

We built a 32-ad golden set from the held-out test set, with ground truth
sourced independently of any model or LLM output. We generated an
explanation for each ad through the real production pipeline, then scored
each explanation by hand against a rubric that was locked before any
explanation was generated.

| Axis | Mean (0 to 2) | Failure rate (score of 0) |
|---|---|---|
| Groundedness | 1.656 | 6.2% (2 of 32) |
| Rule accuracy | 1.656 | 9.4% (3 of 32) |
| Hallucination | 1.438 | 25.0% (8 of 32), 95% CI 13.3% to 42.1% |

The hallucination rate is not evenly distributed. It is 75% (3 of 4) on
ads the classifier misclassified, and 17.9% (5 of 28) on ads the
classifier correctly classified. We traced 7 of the 8 major
hallucinations to one specific, ambiguous sentence in the
`mismatched_creative` policy rule's own text, verified by re-running the
actual retrieval call for every hallucinating example, not by assumption.

| Group | Count | Hallucination failure rate |
|---|---|---|
| Misclassified ads | 4 | 75% (3 of 4) |
| True-positive ads | 28 | 17.9% (5 of 28) |

Full detail: `eval/README.md`, `eval/root_cause_analysis.md`, and
`eval/results.json`.

## 3. Guardrails: prompt injection testing

We built 8 adversarial `ad_text` examples, covering instruction override,
fake system-role injection, topic hijack, system-prompt extraction, fake
authority override, fake retrieved-context injection, roleplay persona
reframing, and fake conversation history. We ran all 8 against the
pipeline before any defense existed.

| Test phase | Result |
|---|---|
| Baseline (no defense) | 8 of 8 adversarial cases resisted |
| After defense (system prompt hardening + output validator added) | 8 of 8 still resisted, no regression |
| Output validator false-positive check | 0 of 32 real Phase 8 explanations incorrectly flagged |
| Output validator detection check | 4 of 4 synthetic already-successful-injection strings caught, after fixing a real bug found during this check |

The honest finding here is that the baseline already resisted every
tested attack, with zero task-specific hardening in place. We added
defense-in-depth anyway, since relying solely on undocumented model
behavior is not a defensible engineering position. "8 of 8 resisted" is
evidence of robustness against known attack categories on one model. It
is not a security guarantee, and it is not the same claim as exhaustive
red-teaming.

Full detail: `guardrails/README.md`.

## 4. Agent reproducibility: cross-ad conflation

We built an agent that autonomously orchestrates the Phase 6 MCP tool
across an 18-ad batch. We verified structurally that its tool calls come
from the model's own decisions, not from a scripted loop. We then re-ran
the identical batch 3 times, to check whether an initially observed
cross-ad error in the agent's own synthesis was a one-off or a
reproducible pattern.

| Run | Section 1 category-table errors | Section 3 confidence-table errors | Narrative-only errors |
|---|---|---|---|
| 1 | 0 | 0 | 1 |
| 2 | 2 | 3 | 0 |
| 3 | 2 | 7 (over half the 14-row table) | 0 |

Conflation occurred in 3 of 3 runs, which makes it a reproducible failure
mode, not a sampling fluke. Its severity is not stable or predictable
from run to run: it ranges from one misattributed word to corrupting over
half of a results table. The underlying classifier is fully deterministic
across all three runs, confirmed directly, which rules out model
non-determinism as the explanation. The variation is entirely in the
agent's own synthesis of already-correct, already-stable data.

Full detail: `agent/README.md` and `agent/results/conflation_check.json`.

## Summary table

| Verification area | Headline result | Confidence in the result |
|---|---|---|
| Ablation study | Joint model beats both single modalities, with non-overlapping confidence intervals | High. Verified on the same fixed split, cross-checked against theoretical expectations. |
| LLM hallucination rate | 25.0%, with root cause traced to one specific rule sentence | High on the rate itself, with an honestly wide confidence interval given the small sample. Root cause verified by re-running actual retrieval calls. |
| Guardrails | 8 of 8 known attacks resisted, both before and after hardening | Moderate. This covers known attack categories on one model, not exhaustive red-teaming. |
| Agent reproducibility | Cross-ad conflation confirmed in 3 of 3 runs, severity unpredictable | High on reproducibility, low on severity distribution, since only 3 runs were performed. |

Every limitation named above is also stated in the relevant phase's own
README, and in the top-level `README.md`'s "Limitations" section. Nothing
here is presented as more certain than the underlying evidence supports.
