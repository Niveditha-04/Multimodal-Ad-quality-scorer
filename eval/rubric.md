# Explanation scoring rubric

This rubric was written and locked before generating or scoring any
explanations. The golden set's ground-truth labels were written next,
independently of this rubric's wording. Explanations were generated only
after both were finalized. This file does not get adjusted once scoring
starts, no matter what the generated explanations look like. If a real
ambiguity comes up during scoring that this rubric does not cleanly
handle, it gets logged as a note on that specific example. It is never
used to retroactively rewrite the criteria.

There are three axes. Each is scored 0, 1, or 2 per example.

## 1. Groundedness

This axis asks: does the explanation correctly describe what is actually
shown in the image? This includes the species, the visible condition or
quality, and the setting.

| Score | Label | Meaning |
|---|---|---|
| 2 | Accurate | Every visual claim the explanation makes about the image is correct. |
| 1 | Vague or partially accurate | The explanation describes the image in terms too general to be wrong (for example, it never actually describes what is pictured). Or it gets a general category right, such as "an animal," but gets a specific checkable detail wrong. |
| 0 | Inaccurate | The explanation makes a specific, checkable visual claim about the image that is factually wrong. This includes wrong species, wrong condition, or a false claim that something is present or absent in the image. |

## 2. Rule accuracy

This axis asks: does the explanation cite the policy rule, from
`db/policy_categories.py`, that genuinely applies? This is judged against
the ad's actual ground truth, not against what the classifier predicted.

| Score | Label | Meaning |
|---|---|---|
| 2 | Correct rule, correctly applies | The explanation names the specific rule that actually governs this ad's real violation. For a genuinely compliant ad, it correctly asserts no rule violation. |
| 1 | Related but imprecise | The explanation cites a rule in the right general category but not the exact one that applies. Or its reasoning for why the cited rule applies is muddled or hedged rather than clearly wrong. |
| 0 | Wrong or fabricated rule | The explanation cites a rule that does not actually apply to this ad. Or it invents a requirement that is not in any of the 7 real policy rules, such as a "the product must be visible in the image" rule that does not exist. |

**Special case: the ground truth is `approved`, meaning the ad was wrongly
flagged.** In this case, no violation rule genuinely applies, by
definition, since the ad is actually compliant. Any explanation that
asserts a specific rule violation on a genuinely compliant ad scores 0 on
this axis automatically. This holds no matter how confidently or fluently
the explanation is argued. There is no partial credit for citing "a rule
that would apply if the classifier's prediction were correct." This is
deliberate. The whole point of this axis is whether the rule genuinely
applies to reality, not whether the explanation is internally consistent
with the model's own possibly-wrong prediction.

## 3. Hallucination

This axis asks: does the explanation assert any specific, checkable claim
that is not actually verifiable from the ad's real image and text?

| Score | Label | Meaning |
|---|---|---|
| 2 | No hallucination | Every specific factual claim in the explanation is directly verifiable from the ad text or the image. |
| 1 | Minor unverifiable claim | The explanation makes one vague inference stated with more confidence than it has earned. This claim is not central to the stated reasoning for why the ad was flagged. |
| 0 | Major hallucination | The explanation asserts a specific, checkable claim, about the image, the ad text, or the policy, that is fabricated. This claim is central to the explanation's stated reasoning. An example is describing an object as absent from an image when the image was never correctly described at all. |

Groundedness and hallucination often move together, but they are not the
same axis. Groundedness asks whether the description of the image is
accurate. Hallucination asks whether the explanation asserted something
specific and unverifiable as fact. A wrong species claim fails both axes.
A confident claim about why a rule applies, when that claim is not
actually supported by anything in the ad or the rule text, such as an
invented "product must be visible in frame" requirement, is primarily a
hallucination issue. This is true even if the explanation happens to get
the image's species right.

## Aggregate reporting

For each axis, we report the mean score on a 0 to 2 scale, along with the
count and percentage scoring 0. That count is the axis's "failure rate."
The hallucination rate is reported as the percentage of the golden set
scoring 0 on the hallucination axis specifically. That percentage is the
headline number. It is not an average.
