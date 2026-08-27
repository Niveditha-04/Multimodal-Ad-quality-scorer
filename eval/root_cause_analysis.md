# Root cause of the hallucination pattern: traced to specific rule text

We investigated a specific question. Is the "product must be visibly
depicted in the image" fabrication, described in `eval/README.md`, the LLM
improvising from nothing? Or is it mis-generalizing from something it was
actually given?

The answer is the second one, for 7 of the 8 hallucination cases. We
traced it to a specific, identifiable ambiguity in one policy rule's
actual text. It is not generic model unreliability.

## The source: the wording of the `mismatched_creative` rule

This is the rule's exact text, quoted verbatim from `db/policy_categories.py`
(this quote is not rewritten for style, since changing the wording of a
direct quote would misrepresent what the actual source file says):

> "The product or animal shown in the image must match what the ad copy
> describes. An ad for a small-breed harness must not show a large-breed
> dog; an ad for cat litter must not show a dog. Mismatches between the
> pictured animal/product and the text description mislead the buyer
> about what they're purchasing."

The rule pairs the words "product" and "animal" as if both are things
that should appear in the image and be checked for a match. But every one
of the rule's own worked examples, like "must not show a large-breed
dog," is entirely about the pictured animal. No example in the rule ever
describes a case where the product itself needs to be physically visible.

The literal wording does not rule out the other reading, though. Phrases
like "product or animal shown in the image must match" and "pictured
animal or product" leave the door open. When an LLM is asked to justify a
`mismatched_creative` violation on an ad where the animal genuinely does
match the text, because the ad is actually compliant or violates a
different rule, it has nothing else in the rule to point to. So it
reaches for the other half of "product or animal" and asserts that the
product is not shown.

## We verified this rule was actually retrieved, not just theoretically available

We re-ran `rag.retrieve.retrieve_context()` using the actual ad text for
every hallucinating example. This let us check what was actually placed
in the prompt.

| Ad IDs | Predicted label | Retrieval path | Was `mismatched_creative` retrieved? |
|---|---|---|---|
| 6, 97, 105, 146 | misleading | Guaranteed by a 1-to-1 label mapping | Yes, by construction |
| 144, 157, 185, 211 | policy_violation | Pure semantic search | Yes. Confirmed by re-running retrieval. |
| 412 | low_quality | Guaranteed rule was `low_quality_creative` | No. Retrieved rules were `low_quality_creative` and `unsubstantiated_health_claims`. Neither mentions image content. |

For the 4 `misleading` cases, `mismatched_creative` is guaranteed by
design in `retrieve.py`. See `rag/retrieve.py`'s
`LABEL_TO_POLICY_CATEGORY` mapping. So its presence there is not new
information.

The genuinely new finding is the 4 `policy_violation` cases.
`mismatched_creative` is not guaranteed for these. Only the
`misleading` and `low_quality` labels get a guaranteed rule. Here, it was
pulled in by pure semantic search instead. We verified its presence in
the prompt directly, by re-running the actual retrieval call. We did not
assume it.

## Result: 7 of 8 major hallucinations trace to this one phrase

The file `eval/scores.py` records 8 examples that scored 0, meaning major
hallucination.

Seven of them (ad IDs 6, 97, 105, 144, 157, 185, and 211) cite
`mismatched_creative` and fabricate a "product not visible"
justification. All 7 had the exact ambiguous rule text in their prompt,
confirmed above.

Ad 146 scored 1, not 0. This means it is a minor or secondary
hallucination, not the central reasoning. It shows the identical pattern
in a weaker form. Its explanation states "no food product is visible
anywhere in the image, creating a disconnect," tacked onto an otherwise
correct explanation. This ad also had the guaranteed `mismatched_creative`
rule in its prompt.

One example, ad 412 (`low_quality`), does not fit this explanation. Its
retrieved rules, `low_quality_creative` and
`unsubstantiated_health_claims`, contain no image-content-matching
language at all. This one case looks like an independent model tendency,
not something traceable to specific retrieved text. We report it honestly
as a separate, less-understood failure mode. We do not fold it into the
same explanation just to make a cleaner-looking story.

## Why this matters more than "the LLM sometimes hallucinates"

This is a fixable, specific defect, not a vague reliability concern. The
rule text itself invites the misreading. A concrete fix, not implemented
here since it is out of scope for this evaluation phase but a natural
next step, would be to rewrite `mismatched_creative`'s description. The
rewrite would explicitly say the rule concerns the pictured animal
matching the ad copy's claimed species or breed, and that the physical
product does not need to appear in the photo. This means removing the
"product or animal... pictured animal or product" phrasing that currently
reads as ambiguous. This is a materially different and more actionable
finding than "explanations sometimes fabricate things." It points at one
specific sentence in one specific file.
