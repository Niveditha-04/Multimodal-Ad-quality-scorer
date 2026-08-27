# Phase 10: Agentic MCP Orchestration, the "Ad Campaign Auditor"

This phase builds an agent that autonomously orchestrates the Phase 6 MCP
tool, `check_ad_compliance`, across a batch of ads. MCP stands for Model
Context Protocol. It is a standard way for an LLM to call external tools.
The agent produces a synthesized review of the whole batch, rather than a
human calling `/score` one ad at a time.

## How this differs from Phase 6

Phase 6 is a single callable tool. Given one image and one ad's text, it
returns one classification and, if the ad is flagged, one explanation. It
has no memory of any other ad. It makes no comparisons. It produces no
synthesis.

Phase 10 is an agent that decides, on its own, which tool calls to make
and in what order. It gathers results across an entire batch and produces
something no single `check_ad_compliance` call could produce alone: cross-ad
pattern detection, a prioritization for human review based on severity and
confidence, and a written rationale for that prioritization. The tool
itself, and the underlying classifier, retrieval, and LLM pipeline behind
it, is completely unchanged from Phase 6. What is new here is the
orchestration layer on top.

## Architecture

`agent/campaign_auditor.py` runs a real Anthropic Messages API tool-use
loop. This is a manual loop, not the SDK's Tool Runner helper. We chose
the manual loop specifically so every tool-execution decision stays
inspectable for verification, described below. Each tool call the model
requests is dispatched through `mcp_server.server.call_tool()`. This is
the actual MCP dispatch path already validated in Phase 6, not a bare
function call that happens to share the implementation. The agent has no
other way to learn about an ad's content except by calling the tool.
Nothing about any specific ad is in its system prompt.

## The batch is drawn mostly from the golden set, on purpose

The batch has 18 ads, built by `agent/build_batch.py` into
`agent/batch.json`. Four are fresh, correctly classified `approved` ads,
so the batch has real "no violation" outcomes, which the golden set alone
does not have. The other 14 are pulled from `eval/golden_set.json`: all 4
of this test set's genuine misclassification cases, and 10 true-positive
violations spread across the three categories.

Using the golden set means every claim the agent's final summary makes
about a given ad can be checked against ground truth that was already
independently verified in Phase 8. That ground truth includes species and
breed, sourced from the Oxford Pet filename convention, and the genuine
policy rule, sourced by matching against the actual violation templates.
None of it is just assumed correct. Given what Phase 8 found about this
pipeline's tendency toward fluent, confident fabrication, an agentic
summary layered on top of the same pipeline needed the same scrutiny, not
less.

**`ad_id` means two different things depending on which file you are
reading, and this is not called out anywhere else.** In `agent/batch.json`
and `eval/golden_set.json`, `ad_id` is a 1-indexed row number into
`data/dataset.csv`, the fixed 480-row source table. But every actual tool
call made through `mcp_server.server.call_tool()` inserts a new row into
the live `ads` database table and returns that row's own auto-increment
database id instead, which is a different number. For example, batch
`ad_id: 6` comes back from the tool with database `ad_id: 515`. The code
already handles this correctly: `agent/verify_conflation.py` keys off the
tool-call log, not the batch's row number, so results are not affected.
But anyone extending this batch or writing a new script against these
files should not assume `ad_id` means the same thing in both places.

## Verifying the agent is actually agentic, not a scripted loop

We checked this structurally. We did not assume it. The loop in
`agent/campaign_auditor.py` is generic. It simply keeps going while
`stop_reason == "tool_use"`. It never iterates over the batch itself to
force calls. Every tool call comes from `tool_use` blocks inside Claude's
own response content. We confirmed the following from the saved
transcript, `agent/results/audit_run.json`.

The run took 2 loop iterations total. In iteration 1, given the prompt
"audit this batch of 18 ad IDs: [...]", the model's single response
contained 18 parallel `tool_use` blocks. It decided, in one turn, to call
the tool once for every ad in the list. In iteration 2, after receiving
all 18 `tool_result` blocks, the model's response contained no further
tool calls. Its `stop_reason` was `"end_turn"`, and it produced the final
text summary instead.

All 18 tool calls target unique ad IDs that exactly match the batch. The
model read the list it was given and acted on it. It did not loop a
fixed count or guess IDs. The system prompt never specifies a call count,
an order, or any ad content. The decision of what to call and when came
entirely from the model parsing the batch list in the first user message.

## A real bug caught mid-phase: silent truncation

The first full run's final response had `stop_reason: "max_tokens"`. This
means it was cut off at the 2000-token cap, not naturally finished. It
happened to end right after a plausible-looking table row. This would
have let the response slip through as "the agent's complete summary" if
we had not checked `stop_reason` explicitly, instead of just reading the
text and deciding it looked done. We preserved this run as evidence, in
`agent/results/audit_run_truncated_bug_example.json`. We fixed the issue
by raising `MAX_TOKENS` to 8000 and adding an explicit
`stop_reason != "end_turn"` check to `main()`, so a truncated run cannot
silently pass as complete again. We re-ran the full audit. The current
`agent/results/audit_run.json` has `stop_reason: "end_turn"`, confirmed
directly, not assumed.

## The actual summaries, and how accurate they actually are

The full text of all three runs is saved. Run 1, the originally analyzed
run, is in `agent/results/audit_run.json`. Two more runs on the identical
18-ad batch are in `audit_run_2.json` and `audit_run_3.json`. We added
these two runs specifically to check whether an initially observed
cross-ad error was a one-off or a real pattern. See Finding 1 below. We
ran these extra checks per explicit instruction, so a single-instance
observation would not get locked into this README as a confirmed pattern.

**A correction to how this section originally read.** The first version
of this README, written after only run 1, said that all 18 predicted
labels and all 14 confidence values in the priority table matched the
underlying data exactly, with zero numeric fabrication. That sentence was
true, and it was checked, but it turned out not to be representative.
Runs 2 and 3 show real numeric and categorical fabrication in the
structured tables. Run 1 simply did not happen to produce any. This is
exactly why the reproducibility check below was worth doing before
treating one clean run as the pipeline's normal behavior.

### Finding 1: the agent's own synthesis introduces cross-ad conflation errors, and it is reproducible

This is the standout finding of this phase. It is a different, higher-level
failure mode than anything in Phase 8. Phase 8 found hallucinations inside
individual explanations: one LLM call inventing a rule, or misreading one
ad's content. This finding is about the agent's own cross-ad synthesis
step mixing up facts between two entirely separate, individually correct
tool outputs. It is an error introduced at the orchestration layer, even
when every underlying tool call was accurate.

**How this was checked.** We re-ran the identical 18-ad batch two more
times, saved as `agent/results/audit_run_2.json` and `audit_run_3.json`.
We then verified each run's summary against its own raw `tool_call_log`
programmatically, using `agent/verify_conflation.py`. This script parses
actual markdown table rows, rather than guessing based on proximity. A
first, cruder version of the script produced false positives from numbers
that happened to appear near an ad ID in unrelated sentences. Results are
saved to `agent/results/conflation_check.json`.

**Result: conflation occurred in 3 of 3 runs, with sharply varying
severity.**

| Run | Section 1 category-table errors | Section 3 confidence-table errors | Narrative-only errors |
|---|---|---|---|
| 1 | 0 | 0 | 1. Ad 146 was described as advertising a "cat carrier." That is actually ad 161's product. Ad 146's real product is cat food. |
| 2 | 2. Ads 161 and 243 were swapped between categories. | 3. Ads 161, 178, and 243 had their confidence values cross-wired with each other. Even a visible self-correction mid-table substituted one wrong borrowed number for another. | 0 |
| 3 | 2. Ads 367 and 243 were swapped between categories. | 7. Ads 105, 161, 178, 243, 359, 367, and 388, over half of the 14-row table, show confidence values borrowed from a different ad in the same batch. | 0 |

Run 1's error was a single mistaken word in one sentence. Run 3's errors
corrupted more than half of the priority table's confidence column.
Several of them look like chained misattributions. For example, ad 367's
real confidence value appears in ad 388's row, while ad 367's own row
shows ad 359's real value instead.

The classifier itself is fully deterministic across all three runs. Every
raw `tool_call_log` entry for a given ad ID has the identical
`predicted_label` and `confidence` in all three runs. We confirmed this
directly. It rules out model non-determinism as the explanation. The
variation is entirely in the agent's own synthesis of already correct,
already stable data.

**The correct claim, and the one to actually use, is this.** This is a
reproducible failure mode of the batch-synthesis step. It occurred in 3
of 3 runs. It is not a sampling fluke. But its severity is not stable or
predictable from run to run. It ranges from a single misattributed word
to corrupting most of a results table. Anyone relying on this agent's
output to actually drive a review queue would need independent
verification of the structured data, not just trust in the narrative.
That is the real, honest conclusion. It is stronger and more defensible
than "observed once."

### Finding 2: the agent inherits Phase 8's known hallucination without correcting for it

Three of the four misclassified ads in this batch, ad IDs 97, 105, and
146, reproduced the same "product must be visible in the image"
fabrication documented in `eval/root_cause_analysis.md`. This was
expected, since it is the same underlying `rag/explain.py` call. The
agent's Section 2 groups these together with the genuine mismatches, ad
IDs 243, 300, and 359, as one undifferentiated pattern. It presents
fabricated and real violations with equal confidence. The agent has no
way to know that some of its "mismatch" examples are actually compliant
ads the classifier got wrong. It takes tool output as ground truth, which
is correct behavior given the information it has. But this means an agent
layered on top of an imperfect pipeline does not self-correct that
pipeline's errors.

### Finding 3: a different hallucination type than anything in Phase 8

Ad 6's explanation was generated fresh during run 1. This was a new API
call, not reused from Phase 8. It did not reproduce the "product must be
visible" pattern. Instead, it fabricated something new: a confident,
specific misidentification of the pictured cat's breed. It asserted
"Ragdoll cat... identifiable by its blue eyes, colorpoint coat, and long,
fluffy fur." Ground truth says the cat is a Birman, and a direct
re-inspection of the image, specifically for the features that
distinguish Birman from Ragdoll, agrees. The cat has the white-gloved
paws that are Birman's signature trait. The "evidence" the model cited
are traits common to both breeds, and do not actually distinguish between
them. This shows that the underlying error, fabricating some specific
justification to support a wrong verdict, is reproducible across
independent LLM calls, even when the specific fabricated content varies
by sampling.

### Finding 4: prioritization logic is sound in principle, inconsistent in practice across runs

The confidence-based reasoning is genuinely useful in isolation. On run
1, ranking primarily by low confidence correctly surfaced 2 of this
batch's 4 real misclassifications, ads 146 and 105, in the top 3
review-priority slots. But the three runs used different strategies to
combine the instruction's two stated factors, severity and confidence.
Run 1 ranked primarily by confidence, which surfaced the actually-wrong
ad 146 at rank 1 overall. Run 2 ranked primarily by category severity,
with confidence only used as a tiebreaker within a tier. This pushed the
genuinely wrong ad 146 down to rank 5, behind several `policy_violation`
ads the model was already correctly and confidently right about. All of
these are defensible readings of "combine confidence and severity," but
they produce materially different practical priority lists from run to
run. This is worth knowing if this agent were ever used to actually drive
a review queue, rather than just demonstrated once. Given Finding 1, this
instability is now a secondary concern, next to whether the table's
numbers can be trusted at all in a given run.

## Reproducing this

```bash
python -m agent.build_batch          # writes agent/batch.json
python -m agent.campaign_auditor [output_path]   # runs the full audit, defaults to agent/results/audit_run.json
python -m agent.verify_conflation    # checks a run's summary against its own raw tool_call_log for cross-ad conflation
```

`campaign_auditor.py` accepts an optional output path argument. We used
this to run the same batch multiple times without overwriting prior
evidence. The files `audit_run.json`, `audit_run_2.json`, and
`audit_run_3.json` are all real, independently generated runs on the
identical batch. None of them is a variation of one run edited
afterward.

## Limitations

Three runs is enough to call the cross-ad conflation in Finding 1 a real,
reproducible failure mode, rather than a single sampling fluke. It is not
enough to characterize its severity distribution precisely. We do not
know whether it usually averages out to something minor, or whether run
3's extensive corruption represents a real, non-trivial share of runs. A
rigorous answer would need many more runs, which we did not do here.

The agent's tool-use pattern here, one parallel burst of calls followed by
one synthesis turn, is the natural shape for this specific task size: 18
independent, order-independent lookups. A batch requiring genuinely
sequential reasoning, for example "check ad A, and based on what you
find, decide whether ad B needs a different kind of check," would
exercise the loop's multi-iteration path differently. We did not test
that case here, since this task does not have that structure.

There was no independent second scoring pass on the agent's summary
itself, unlike Phase 8's rubric-scored golden set. The accuracy claims
here come from direct cross-referencing against ground truth and raw tool
output, not from a formal per-claim rubric.

The table parser in `verify_conflation.py` is specific to the markdown
table shapes actually produced across these 3 runs. It is not a general
markdown-table parser. A summary formatted differently, from a future run
or a different model, could produce table rows it fails to parse
correctly. We cross-checked its counts against an independent manual read
of all three summaries before trusting it. A first, cruder version of
this same script produced false positives from proximity-matching,
rather than actually parsing table structure. We caught this before
relying on its output, the same discipline applied to the
output-validator check in Phase 9.
