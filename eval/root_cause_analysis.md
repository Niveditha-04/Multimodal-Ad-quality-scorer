# Root cause of the hallucination pattern: traced to specific rule text

Investigated whether the "product must be visibly depicted in the image"
fabrication (see `eval/README.md`) is the LLM improvising from nothing, or
mis-generalizing from something it was actually given. It's the latter, for
7 of the 8 hallucination cases -- traced to a specific, identifiable
ambiguity in one rule's actual text, not generic model unreliability.

## The source: `mismatched_creative`'s own wording

From `db/policy_categories.py`:

> "**The product or animal** shown in the image must match what the ad copy
> describes. An ad for a small-breed harness must not show a large-breed
> dog; an ad for cat litter must not show a dog. Mismatches between the
> **pictured animal/product** and the text description mislead the buyer
> about what they're purchasing."

The rule pairs "product" and "animal" as if both are things that should
appear in the image and be checked for a match. Its own examples ("must not
show a large-breed dog") are entirely about the pictured *animal* -- no
example in the rule ever describes a case where the product itself needs to
be physically visible. But the literal wording ("product **or** animal
shown in the image must match," "pictured animal**/product**") doesn't rule
that reading out, and an LLM asked to justify a mismatched_creative
violation on an ad where the animal genuinely does match the text (because
the ad is actually compliant, or violates a different rule) has nothing
else in the rule to point to -- so it reaches for the other half of "product
or animal" and asserts the product isn't shown.

## Verified this rule was actually retrieved, not just theoretically available

Re-ran `rag.retrieve.retrieve_context()` with the actual ad text for every
hallucinating example and checked what was actually in the prompt:

| ad_id | predicted_label | retrieval path | mismatched_creative retrieved? |
|---|---|---|---|
| 6, 97, 105, 146 | misleading | guaranteed (1:1 label mapping) | yes, by construction |
| 144, 157, 185, 211 | policy_violation | pure semantic search | **yes** -- confirmed by re-running retrieval |
| 412 | low_quality | guaranteed (`low_quality_creative`) | no -- retrieved `low_quality_creative` + `unsubstantiated_health_claims`, neither mentions image content |

For the 4 `misleading` cases, `mismatched_creative` is guaranteed by
`retrieve.py`'s design (see `rag/retrieve.py`'s `LABEL_TO_POLICY_CATEGORY`),
so its presence isn't new information. The genuinely new finding is the 4
`policy_violation` cases: `mismatched_creative` was **not** guaranteed for
these (only `misleading`/`low_quality` get a guaranteed rule) -- it was
pulled in by pure semantic search, and its presence in the prompt was
verified directly by re-running the actual retrieval call, not assumed.

## Result: 7 of 8 major hallucinations trace to this one phrase

Of the 8 examples scoring 0 (major hallucination) in `eval/scores.py`:

- **7** (6, 97, 105, 144, 157, 185, 211) cite `mismatched_creative` and
  fabricate a "product not visible" justification -- all 7 had the exact
  ambiguous rule text in their prompt, confirmed above.
- Ad 146 (scored 1, not 0 -- a minor/secondary hallucination, not the
  central reasoning) shows the identical pattern in weaker form: "no food
  product is visible anywhere in the image, creating a disconnect,"
  tacked onto an otherwise-correct explanation. Also had the guaranteed
  `mismatched_creative` rule in its prompt.
- **1** (ad 412, `low_quality`) does not fit this explanation. Its
  retrieved rules (`low_quality_creative`, `unsubstantiated_health_claims`)
  contain no image-content-matching language at all. This one looks like
  an independent model tendency, not traceable to specific retrieved text --
  reported honestly as a different, less-understood failure mode rather
  than folded into the same explanation for a cleaner-looking story.

## Why this matters more than "the LLM sometimes hallucinates"

This is a fixable, specific defect, not a vague reliability concern: the
rule text itself invites the misreading. A concrete fix (not implemented
here, out of scope for this evaluation phase, but the natural next step)
would be rewriting `mismatched_creative`'s description to explicitly say
the rule concerns the pictured *animal* matching the ad copy's claimed
species/breed, and that the physical product does not need to appear in
the photo -- removing the "product or animal... pictured animal/product"
phrasing that currently reads as ambiguous. This is a materially different
and more actionable finding than "explanations sometimes fabricate
things": it points at one sentence in one file.
