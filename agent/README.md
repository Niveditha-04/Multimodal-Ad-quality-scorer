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

## The actual summaries, and how accurate they actually are

Full text of all three runs: `agent/results/audit_run.json` (run 1, the
originally analyzed run), `audit_run_2.json`, `audit_run_3.json` (2
additional runs on the identical 18-ad batch, added specifically to check
whether an initially-observed cross-ad error was a one-off or a real
pattern -- see Finding 1 below; run per explicit instruction not to lock a
single-instance observation into this README as a confirmed pattern).

**Important correction to how this section originally read**: the first
version of this README, written after only run 1, said "all 18 predicted
labels and all 14 confidence values in the priority table match the
underlying data exactly... zero numeric fabrication." That sentence was
true, checked, and not fabricated -- but it turned out not to be
representative. Runs 2 and 3 show real numeric and categorical fabrication
in the structured tables, which run 1 simply didn't happen to produce. This
is exactly why the reproducibility check below was worth doing before
treating one clean run as the pipeline's normal behavior.

### Finding 1 (the standout finding of this phase): the agent's own synthesis introduces cross-ad conflation errors, and it's reproducible

This is a different, higher-level failure mode than anything in Phase 8:
Phase 8 found hallucinations *inside* individual explanations (one LLM call
inventing a rule or misreading one ad's content). This is the agent's own
cross-ad synthesis step mixing up facts *between* two entirely separate,
individually-correct tool outputs -- an error introduced at the
orchestration layer even when every underlying tool call was accurate.

**How this was checked**: re-ran the identical 18-ad batch two more times
(`agent/results/audit_run_2.json`, `audit_run_3.json`), then verified each
run's summary against its own raw `tool_call_log` programmatically
(`agent/verify_conflation.py` -- parses actual markdown table rows rather
than proximity-guessing, after a first, cruder version of the script
produced false positives from numbers appearing near an ad_id in unrelated
sentences; results saved to `agent/results/conflation_check.json`).

**Result: conflation occurred in 3 of 3 runs, with sharply varying
severity:**

| Run | Section-1 category-table errors | Section-3 confidence-table errors | Narrative-only errors |
|---|---|---|---|
| 1 | 0 | 0 | 1 (Ad 146 described as advertising a "cat carrier" -- that's ad 161's product; 146's real product is cat food) |
| 2 | 2 (ads 161, 243 swapped between categories) | 3 (ads 161, 178, 243's confidence values cross-wired with each other -- even a visible self-correction mid-table substituted one wrong borrowed number for another) | 0 |
| 3 | 2 (ads 367, 243 swapped between categories) | 7 (ads 105, 161, 178, 243, 359, 367, 388 -- over half the 14-row table -- show confidence values borrowed from a different ad in the same batch) | 0 |

Run 1's error was a single mistaken word in one sentence. Run 3's errors
corrupted more than half of the priority table's confidence column, several
via what look like chained misattributions (e.g. ad 367's real confidence
value appears in ad 388's row, while ad 367's own row shows ad 359's real
value). **The classifier itself is fully deterministic across all three
runs** -- every raw `tool_call_log` entry for a given ad_id has the
identical `predicted_label` and `confidence` in all three runs, confirmed
directly, which rules out model non-determinism as the explanation. The
variation is entirely in the agent's own synthesis of already-correct,
already-stable data.

**Correct claim, and the one to actually use**: this is a reproducible
failure mode of the batch-synthesis step (occurred in 3/3 runs), not a
sampling fluke -- but its *severity* is not stable or predictable run to
run, ranging from a single misattributed word to corrupting most of a
results table. Anyone relying on this agent's output to actually drive a
review queue would need independent verification of the structured
data, not just trust in the narrative. That's the real, honest conclusion,
and a stronger and more defensible one than "observed once."

### Finding 2: the agent inherits Phase 8's known hallucination without correcting for it

Three of the four misclassified ads in this batch (97, 105, 146) reproduced
the same "product must be visible in the image" fabrication documented in
`eval/root_cause_analysis.md` -- expected, since it's the same underlying
`rag/explain.py` call. The agent's Section 2 groups these together with the
*genuine* mismatches (243, 300, 359) as one undifferentiated pattern,
presenting fabricated and real violations with equal confidence. The agent
has no way to know some of its "mismatch" examples are actually compliant
ads the classifier got wrong -- it takes tool output as ground truth, which
is correct behavior *given its information*, but means an agent layered on
top of an imperfect pipeline does not self-correct that pipeline's errors.

### Finding 3: a different hallucination *type* than anything in Phase 8

Ad 6's explanation, generated fresh during run 1 (a new API call, not
reused from Phase 8), didn't reproduce the "product must be visible"
pattern -- it fabricated something new: a confident, specific
misidentification of the pictured cat's breed (asserting "Ragdoll cat...
identifiable by its blue eyes, colorpoint coat, and long, fluffy fur" when
ground truth, and a direct re-inspection of the image specifically for
Birman-vs-Ragdoll distinguishing features, both indicate Birman -- the cat
has the white-gloved paws that are Birman's signature trait, while the
"evidence" the model cited are traits common to both breeds and don't
actually distinguish between them). This shows the underlying error
(fabricating *some* specific justification to support a wrong verdict) is
reproducible across independent LLM calls even when the *specific*
fabricated content varies by sampling.

### Finding 4: prioritization logic is sound in principle, inconsistent in practice across runs

The confidence-based reasoning is genuinely useful in isolation: on run 1,
ranking primarily by low confidence correctly surfaced 2 of this batch's 4
real misclassifications (ads 146 and 105) in the top 3 review-priority
slots. But the three runs used different strategies to combine the
instruction's two stated factors (severity and confidence): run 1 ranked
primarily by confidence (surfacing the actually-wrong ad 146 at #1
overall); run 2 ranked primarily by category severity, with confidence
only a tiebreaker within a tier -- which pushed the genuinely-wrong ad 146
down to #5, behind several `policy_violation` ads the model was already
correctly and confidently right about. All are defensible readings of
"combine confidence and severity," but they produce materially different
practical priority lists from run to run -- worth knowing if this were
ever used to actually drive a review queue, not just demonstrated once.
Given Finding 1, this instability is now a secondary concern next to
whether the table's *numbers* can be trusted at all in a given run.

## Reproducing this

```bash
python -m agent.build_batch          # writes agent/batch.json
python -m agent.campaign_auditor [output_path]   # runs the full audit; defaults to agent/results/audit_run.json
python -m agent.verify_conflation    # checks a run's summary against its own raw tool_call_log for cross-ad conflation
```

`campaign_auditor.py` accepts an optional output path argument, used here
to run the same batch multiple times without overwriting prior evidence
(`audit_run.json`, `audit_run_2.json`, `audit_run_3.json` are all real,
independently generated runs on the identical batch, not variations of one
run edited afterward).

## Limitations

- Three runs is enough to call the cross-ad conflation in Finding 1 a real,
  reproducible failure mode rather than a single sampling fluke -- it isn't
  enough to characterize its severity distribution precisely (does it
  average out to "usually minor," or does run 3's extensive corruption
  represent a real, non-trivial share of runs?). A rigorous answer would
  need many more runs, which wasn't done here.
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
- `verify_conflation.py`'s table parser is specific to the markdown table
  shapes actually produced across these 3 runs -- it isn't a general
  markdown-table parser, and a summary formatted differently (a future run,
  or a different model) could produce table rows it fails to parse
  correctly. Its counts were cross-checked against an independent manual
  read of all three summaries before being trusted (a first, cruder version
  of this same script produced false positives from proximity-matching
  rather than actually parsing table structure -- caught before relying on
  its output, the same discipline applied to the output-validator check in
  Phase 9).
