# Explanation scoring rubric

Written and locked before generating or scoring any explanations. The golden
set's ground-truth labels are written next, independently of this rubric's
wording, and explanations are generated after both are finalized -- this
file does not get adjusted once scoring starts, regardless of what the
generated explanations look like. If a real ambiguity turns up during
scoring that this rubric doesn't cleanly handle, it gets logged as a
scoring note on that specific example, not used to retroactively rewrite
the criteria.

Three axes, each scored 0/1/2 per example.

## 1. Groundedness

Does the explanation correctly describe what's actually depicted in the
image (species, visible condition/quality, setting)?

- **2 -- Accurate**: every visual claim the explanation makes about the
  image is correct.
- **1 -- Vague or partially accurate**: describes the image in terms too
  general to be wrong (e.g. never actually describes what's pictured) --
  or gets a general category right (e.g. "an animal") but a specific,
  checkable detail wrong.
- **0 -- Inaccurate**: makes a specific, checkable visual claim about the
  image that is factually wrong (wrong species, wrong condition, a claim
  about something being present/absent in the image that isn't true).

## 2. Rule accuracy

Does the explanation cite the policy rule (from `db/policy_categories.py`)
that genuinely applies, given the ad's actual ground truth -- not given
what the classifier predicted?

- **2 -- Correct rule, correctly applies**: names the specific rule that
  actually governs this ad's real violation (or, for a genuinely
  compliant ad, correctly does not assert any rule violation).
- **1 -- Related but imprecise**: cites a rule in the right neighborhood
  (e.g. right general category) but not the exact one that applies, or
  the reasoning for why the cited rule applies is muddled/hedged rather
  than clearly wrong.
- **0 -- Wrong or fabricated rule**: cites a rule that doesn't actually
  apply to this ad, or invents a requirement not present in the actual
  7 policy rules (e.g. a "product must be visible in the image" rule that
  doesn't exist).

**Special case -- ground truth is `approved` (the ad was wrongly flagged):**
no violation rule genuinely applies, by definition, since the ad is
actually compliant. Any explanation that asserts a specific rule violation
on a genuinely-compliant ad scores **0** on this axis automatically,
regardless of how confidently or fluently it's argued -- there is no
partial credit for citing "a rule that would apply if the classifier's
prediction were correct." This is deliberate: the whole point of this axis
is whether the rule genuinely applies to reality, not whether it's
internally consistent with the model's own (possibly wrong) prediction.

## 3. Hallucination

Does the explanation assert any specific, checkable claim that isn't
actually verifiable from the ad's real image and text?

- **2 -- No hallucination**: every specific factual claim in the
  explanation is directly verifiable from the ad text and/or image.
- **1 -- Minor unverifiable claim**: one vague inference stated with more
  confidence than it's earned, but not central to the stated reasoning for
  why the ad was flagged.
- **0 -- Major hallucination**: asserts a specific, checkable claim
  (about the image, the ad text, or the policy) that is fabricated and is
  central to the explanation's stated reasoning -- e.g. describing an
  object as absent from an image when the actual image was never
  correctly described at all.

Groundedness and hallucination often move together but aren't the same
axis: groundedness asks "is the description of the image accurate,"
hallucination asks "did it assert something specific and unverifiable as
fact." A wrong species claim hits both. A confident claim about *why* a
rule applies that isn't actually supported by anything in the ad or the
rule text (like an invented "product must be visible in frame"
requirement) is primarily a hallucination issue even if it happens to get
the image's species right.

## Aggregate reporting

Per axis: mean score (0-2) and the count/percentage scoring 0 ("failure
rate" for that axis). Hallucination rate is reported as the percentage of
the golden set scoring **0** on the hallucination axis specifically --
that is the headline number, not an average.
