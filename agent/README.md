# Phase 10: Agentic MCP Orchestration -- the "Ad Campaign Auditor"

An agent that autonomously orchestrates the Phase 6 MCP tool
(`check_ad_compliance`) across a batch of ads and produces a synthesized
review, rather than a human calling `/score` one ad at a time.

## How this differs from Phase 6

Phase 6 is a single callable tool: given one image and one ad's text, it
returns one classification and (if flagged) one explanation. It has no
memory of any other ad, makes no comparisons, and produces no synthesis.

Phase 10 is an agent that decides, on its own, which tool calls to make
and in what order, gathers results across an entire batch, and produces
something no single `check_ad_compliance` call could produce alone: cross-ad
pattern detection, a severity/confidence-based prioritization for human
review, and a written rationale for that prioritization. The tool itself
(and the underlying classifier + RAG + LLM pipeline behind it) is completely
unchanged from Phase 6 -- what's new here is the orchestration layer on top.

## Architecture

`agent/campaign_auditor.py` runs a real Anthropic Messages API tool-use loop
(the manual loop, not the SDK's Tool Runner helper, specifically so every
tool-execution decision stays inspectable for verification -- see below).
Each tool call the model requests is dispatched through
`mcp_server.server.call_tool()`, the actual MCP dispatch path already
validated in Phase 6 -- not a bare function call that happens to share the
implementation. The agent has no other way to learn about an ad's content
except by calling the tool; nothing about any ad is in its system prompt.

## The batch: drawn mostly from the golden set, deliberately

18 ads (`agent/build_batch.py` / `agent/batch.json`): 4 fresh
correctly-classified `approved` ads (so the batch has real "no violation"
outcomes, which the golden set alone doesn't), plus 14 ads pulled from
`eval/golden_set.json` -- all 4 of this test set's genuine misclassification
cases, and 10 true-positive violations spread across the three categories.
Using the golden set means every claim the agent's final summary makes
about a given ad can be checked against ground truth already
independently verified in Phase 8 (species/breed from the Oxford Pet
filename convention, genuine policy rule from matching against the actual
violation templates) -- not just assumed correct. Given what Phase 8 found
about this pipeline's tendency toward fluent, confident fabrication, an
agentic summary layered on top of the same pipeline needed the same
scrutiny, not less.

## Verifying the agent is actually agentic, not a scripted loop

Checked structurally, not assumed. `agent/campaign_auditor.py`'s loop is
generic -- "keep going while `stop_reason == 'tool_use'`" -- and never
iterates over the batch itself to force calls; every tool call comes from
`tool_use` blocks inside Claude's own response content. Confirmed from the
saved transcript (`agent/results/audit_run.json`):

- **2 loop iterations total.** Iteration 1: given the prompt "audit this
  batch of 18 ad IDs: [...]", the model's single response contained 18
  parallel `tool_use` blocks -- it decided, in one turn, to call the tool
  once for every ad in the list. Iteration 2: after receiving all 18
  `tool_result` blocks, the model's response contained no further tool
  calls (`stop_reason: "end_turn"`) and instead produced the final text
  summary.
- All 18 tool calls target unique `ad_id`s exactly matching the batch --
  the model read the list it was given and acted on it, rather than
  looping a fixed count or guessing IDs.
- The `SYSTEM_PROMPT` never specifies a call count, an order, or provides
  any ad content -- the decision of what to call and when came entirely
  from the model parsing the batch list in the first user message.

## A real bug caught mid-phase: silent truncation

The first full run's final response had `stop_reason: "max_tokens"` --
cut off at the 2000-token cap, not naturally finished. It happened to end
right after a plausible-looking table row, which would have let this slip
through as "the agent's complete summary" if `stop_reason` hadn't been
checked explicitly rather than just reading the text and deciding it
looked done. Preserved as evidence: `agent/results/audit_run_truncated_bug_example.json`.
Fixed by raising `MAX_TOKENS` to 8000 and adding an explicit
`stop_reason != "end_turn"` check to `main()` so a truncated run can't
silently pass as complete again. Re-ran the full audit; the current
`agent/results/audit_run.json` has `stop_reason: "end_turn"`, confirmed
directly, not assumed.

## The actual summary, and how accurate it actually is

Full text: see `agent/results/audit_run.json`'s `final_summary` field
(reproduced in this phase's commit message and available in full there).
Section 1's counts and Section 3's confidence figures were cross-checked
programmatically against the raw tool outputs in `tool_call_log` --
**all 18 predicted labels and all 14 confidence values in the priority
table match the underlying data exactly, to three decimal places. Zero
numeric fabrication.**

### Finding 1: the agent inherits Phase 8's known hallucination without correcting for it

Three of the four misclassified ads in this batch (97, 105, 146) reproduced
the same "product must be visible in the image" fabrication documented in
`eval/root_cause_analysis.md` -- expected, since it's the same underlying
`rag/explain.py` call. The agent's Section 2 groups these together with the
*genuine* mismatches (243, 300, 359) as one undifferentiated 7-ad pattern
("Species/Product Mismatch"), presenting fabricated and real violations
with equal confidence. The agent has no way to know 3 of its 7 "mismatch"
examples are actually compliant ads the classifier got wrong -- it takes
tool output as ground truth, which is the correct behavior *given its
information*, but means an agent layered on top of an imperfect pipeline
does not self-correct that pipeline's errors. It faithfully reports them
as a confident cross-ad "pattern."

### Finding 2 (new, not seen in Phase 8): a different hallucination *type*

Ad 6's explanation, generated fresh during this phase's first run (a new
API call, not reused from Phase 8), didn't reproduce the "product must be
visible" pattern -- it fabricated something new: a confident, specific
misidentification of the pictured cat's breed (asserting "Ragdoll cat...
identifiable by its blue eyes, colorpoint coat, and long, fluffy fur" when
ground truth, and a direct re-inspection of the image specifically for
Birman-vs-Ragdoll distinguishing features, both indicate Birman -- the cat
has the white-gloved paws that are Birman's signature trait, while the
"evidence" the model cited are traits common to both breeds and don't
actually distinguish between them). This shows the underlying error
(fabricating *some* specific justification to support a wrong verdict) is
reproducible across independent LLM calls even when the *specific*
fabricated content varies by sampling -- a stronger, more general version
of the Phase 8 finding than "this one phrase recurs."

### Finding 3 (new to this phase): the agent's own synthesis introduces a cross-ad conflation error

This is the most interesting finding of Phase 10, because it's not
inherited from any tool call -- it originates in the agent's own writing.
Section 2 states "Ad 146 similarly shows only a cat while advertising a
**cat carrier**." Ad 146's actual product is cat food (confirmed both from
`agent/batch.json`'s ground truth and directly from ad 146's own tool
output, which correctly says "cat food" and "food product" throughout).
The cat-carrier product belongs to **ad 161** -- a different ad in the same
batch, also cat-related and also policy_violation-flagged, whose own tool
output correctly and separately describes a cat carrier. The agent
conflated details between two distinct, correctly-reported tool outputs
while writing its own cross-ad summary. This is a distinct risk from
Finding 1: even if every individual tool call were perfectly accurate, the
synthesis layer itself can introduce new errors by mixing up similar
entries across a batch -- a known failure mode in long-context
summarization, now demonstrated concretely in this pipeline.

### Finding 4: prioritization logic is sound in principle, inconsistent in practice across runs

The confidence-based reasoning is genuinely useful: on the first (truncated
but otherwise complete) run, ranking primarily by low confidence correctly
surfaced 2 of this batch's 4 real misclassifications (ads 146 and 105) in
the top 3 review-priority slots. But the two runs used different strategies
to combine the instruction's two stated factors (severity and confidence):
the first run ranked primarily by confidence (surfacing the actually-wrong
ad 146 at #1 overall); this run ranked primarily by category severity, with
confidence only as a tiebreaker *within* a severity tier -- which pushed ad
146 (genuinely wrong, but categorized as the less-severe "misleading" tier)
down to #5, behind three `policy_violation` ads the model was already very
confident about (and which are, per ground truth, actually correct). Both
orderings are defensible readings of "prioritize by a combination of low
confidence and severity," but they produce materially different practical
priority lists from run to run -- worth knowing if this were ever used to
actually drive a review queue, not just demonstrated once.

## Reproducing this

```bash
python -m agent.build_batch          # writes agent/batch.json
python -m agent.campaign_auditor     # runs the full audit, writes agent/results/audit_run.json
```

## Limitations

- Single batch, single run analyzed in depth (plus the earlier truncated
  run, kept as evidence of the token-limit bug). Run-to-run prompt strategy
  variability (Finding 4) was observed from two runs, not systematically
  characterized across many.
- The agent's tool-use pattern here (one parallel burst of calls, then one
  synthesis turn) is the natural shape for this specific task size (18
  independent, order-independent lookups). A batch requiring genuinely
  sequential reasoning (e.g. "check ad A, and based on what you find,
  decide whether ad B needs a different kind of check") would exercise the
  loop's multi-iteration path differently -- not tested here, since this
  task doesn't have that structure.
- No independent second scoring pass on the agent's summary itself (unlike
  Phase 8's rubric-scored golden set) -- accuracy claims here come from
  direct cross-referencing against ground truth and raw tool output, not a
  formal per-claim rubric.
