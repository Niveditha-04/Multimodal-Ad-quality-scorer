"""Phase 10: the "Ad Campaign Auditor" -- an agent that autonomously
orchestrates the Phase 6 MCP tool (check_ad_compliance) across a batch of
ads, rather than a human calling /score one ad at a time.

This is a genuinely different thing from Phase 6, not a relabeling of it:
Phase 6 is a single callable tool that scores one ad on request. This is an
agent that decides, on its own, which tool calls to make and in what order,
gathers the results across an entire batch, and produces a synthesized
review -- counts, cross-ad patterns, and a prioritized human-review list --
that no single check_ad_compliance call could produce by itself.

Uses the real Anthropic Messages API tool-use loop (not the Tool Runner
helper, to keep every tool-execution decision explicit and inspectable for
verification purposes) and dispatches each tool call through
mcp_server.server.call_tool() -- the actual MCP dispatch path validated in
Phase 6, not a bare function call that happens to share the implementation.
"""
import asyncio
import json
import os

import anthropic
from dotenv import load_dotenv

from mcp_server.server import server as mcp_server

load_dotenv()

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 8000
MAX_LOOP_ITERATIONS = 30

TOOL_DEFINITION = {
    "name": "check_ad_compliance",
    "description": (
        "Scores one ad creative (by ad_id) for policy violations and quality "
        "issues using the multimodal classifier. Returns predicted_label "
        "(approved/policy_violation/misleading/low_quality), confidence "
        "(0-1), and an explanation grounded in the specific policy rule "
        "(present only when the ad was flagged)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ad_id": {"type": "integer", "description": "The ad's ID from the batch you were given."},
        },
        "required": ["ad_id"],
    },
}

SYSTEM_PROMPT = (
    "You are an ad campaign auditor for a pet-products marketplace. You are "
    "given a batch of ad IDs and must check each one for policy compliance "
    "using the check_ad_compliance tool, then produce a structured audit "
    "summary. Call the tool once per ad in the batch -- you must check every "
    "ad given to you, not a sample. Base every claim in your summary "
    "strictly on what the tool actually returned; do not guess at an ad's "
    "content or status without having called the tool for it. When you have "
    "checked every ad, produce a final summary with exactly these sections:\n"
    "1. Counts by predicted category.\n"
    "2. Common violation patterns you notice across the batch (if any).\n"
    "3. A prioritized list of which flagged ads most need human review, and "
    "why -- prioritize by a combination of low model confidence and the "
    "severity of the flagged category, and say your reasoning for the "
    "ordering, not just the ordering itself."
)


def _load_batch() -> dict[int, dict]:
    batch = json.loads(open("agent/batch.json").read())
    return {b["ad_id"]: b for b in batch}


async def _execute_tool_call(ad_id: int, batch_by_id: dict[int, dict]) -> dict:
    if ad_id not in batch_by_id:
        return {"error": f"ad_id {ad_id} is not in the given batch"}
    ad = batch_by_id[ad_id]
    result = await mcp_server.call_tool(
        "check_ad_compliance",
        {"image_path": ad["image_path"], "ad_text": ad["ad_text"]},
    )
    text = next((b.text for b in result.content if b.type == "text"), "{}")
    return json.loads(text)


async def run_audit(ad_ids: list[int]) -> dict:
    batch_by_id = _load_batch()
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    messages = [{
        "role": "user",
        "content": (
            f"Audit this batch of {len(ad_ids)} ad IDs: {ad_ids}. "
            f"Check every one and produce the final summary."
        ),
    }]

    tool_call_log = []  # every actual tool call made, in order, for verification

    for iteration in range(MAX_LOOP_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[TOOL_DEFINITION],
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            final_text = next((b.text for b in response.content if b.type == "text"), "")
            return {
                "final_summary": final_text,
                "tool_call_log": tool_call_log,
                "n_iterations": iteration + 1,
                "stop_reason": response.stop_reason,
            }

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            ad_id = block.input.get("ad_id")
            print(f"  [iteration {iteration+1}] agent calls check_ad_compliance(ad_id={ad_id})")
            result = await _execute_tool_call(ad_id, batch_by_id)
            tool_call_log.append({"ad_id": ad_id, "result": result})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })
        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"agent did not finish within {MAX_LOOP_ITERATIONS} iterations")


def main() -> None:
    batch = json.loads(open("agent/batch.json").read())
    ad_ids = [b["ad_id"] for b in batch]
    print(f"auditing {len(ad_ids)} ads: {ad_ids}\n")

    result = asyncio.run(run_audit(ad_ids))

    if result["stop_reason"] != "end_turn":
        # a response cut off by max_tokens can still look complete by
        # coincidence (ended mid-table, right after a plausible-looking
        # row) -- this happened on a real run of this script, caught only
        # by checking stop_reason explicitly, not by reading the text and
        # deciding it looked finished
        print(f"\n*** WARNING: final response did not end naturally "
              f"(stop_reason={result['stop_reason']!r}) -- summary below may be truncated ***\n")

    print(f"\n{'='*60}")
    print(f"agent made {len(result['tool_call_log'])} tool calls across {result['n_iterations']} loop iterations")
    print(f"{'='*60}\n")
    print(result["final_summary"])

    with open("agent/results/audit_run.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nsaved full transcript to agent/results/audit_run.json")


if __name__ == "__main__":
    main()
