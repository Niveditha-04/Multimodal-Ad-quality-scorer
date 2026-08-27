"""Programmatically checks each agent run's final_summary against the raw,
ground-truth tool_call_log from that same run, looking for cross-ad
conflation: cases where the summary attributes ad X's actual
predicted_label or confidence to a different ad Y in a structured table row.

Built after observing what looked like a cross-ad mix-up in run 1's
narrative text, per explicit instruction not to lock that into the README
as a confirmed pattern from a single observation -- re-ran the agent 2 more
times on the identical batch and checked here, the same discipline that
turned Phase 8's rule-fabrication finding from "happened once" into
"verified independently three times" before it was written up as a pattern.

A first version of this script used naive proximity matching (an ad_id
followed within N characters by some decimal number, anywhere in the
summary text) and produced false positives -- e.g. flagging ad 8 as wrong
because ad 25's correctly-stated confidence happened to appear within 60
characters of "8" in an unrelated closing-note sentence, since the two ads
were mentioned near each other but the number wasn't actually attributed to
ad 8. This version parses actual markdown table rows instead: for each row
that names exactly one ad_id and one confidence-shaped number in the same
row, that's a real, unambiguous attribution, not a proximity guess.
"""
import json
import re

ROW_PATTERN = re.compile(
    r"\|[^|\n]*?\b(\d{1,3})\b[^|\n]*?\|[^|\n]*?\|[^|\n]*?(0\.\d{2,3})[^|\n]*?\|"
)


def parse_priority_table_rows(summary: str) -> dict[int, float]:
    """Returns {ad_id: claimed_confidence} for every table row that
    unambiguously names one ad_id and one confidence value."""
    claimed = {}
    for line in summary.split("\n"):
        if "|" not in line:
            continue
        m = ROW_PATTERN.search(line)
        if not m:
            continue
        ad_id, conf = int(m.group(1)), float(m.group(2))
        # skip rows where the "ad_id" is actually a priority rank number
        # (distinguishable: a real ad_id from this batch, a rank 1-18 could
        # collide, so only trust rows containing a batch ad_id AND where
        # that number isn't also plausibly just "priority N" -- resolved by
        # only keeping the LAST bare integer in the row before the pipe
        # that precedes the confidence, which is the ad_id column in every
        # observed table format here)
        claimed[ad_id] = conf
    return claimed


def parse_category_table(summary: str, known_ad_ids: set[int]) -> dict[int, str]:
    """Returns {ad_id: claimed_category} from Section-1-style rows that list
    a label name followed by a comma-separated list of ad_ids."""
    claimed = {}
    label_row = re.compile(
        r"(approved|misleading|policy.?violation|low.?quality)[^\n|]*\|[^\n|]*\|([\d,\s]+)\|",
        re.IGNORECASE,
    )
    for m in label_row.finditer(summary):
        label = m.group(1).lower().replace(" ", "_").replace("-", "_")
        for id_str in re.findall(r"\d+", m.group(2)):
            ad_id = int(id_str)
            if ad_id in known_ad_ids:
                claimed[ad_id] = label
    return claimed


def check_run(path: str) -> dict:
    r = json.loads(open(path).read())
    ground_truth = {c["ad_id"]: c["result"] for c in r["tool_call_log"]}
    summary = r["final_summary"]

    conf_errors = []
    claimed_conf = parse_priority_table_rows(summary)
    for ad_id, claimed in claimed_conf.items():
        if ad_id not in ground_truth:
            continue
        real = ground_truth[ad_id]["confidence"]
        if abs(claimed - real) > 0.01:
            matches = [oid for oid, ot in ground_truth.items()
                       if oid != ad_id and abs(ot["confidence"] - claimed) < 0.005]
            conf_errors.append({"ad_id": ad_id, "real_confidence": round(real, 3),
                                 "claimed_confidence": claimed, "matches_other_ad": matches})

    cat_errors = []
    claimed_cat = parse_category_table(summary, set(ground_truth.keys()))
    for ad_id, claimed in claimed_cat.items():
        real = ground_truth[ad_id]["predicted_label"].replace("_", "")
        if claimed.replace("_", "") != real:
            cat_errors.append({"ad_id": ad_id, "real_label": ground_truth[ad_id]["predicted_label"],
                                "claimed_label": claimed})

    return {"path": path, "category_table_errors": cat_errors, "priority_table_errors": conf_errors}


def main():
    results = []
    for path in ["agent/results/audit_run.json", "agent/results/audit_run_2.json", "agent/results/audit_run_3.json"]:
        result = check_run(path)
        results.append(result)
        n = len(result["category_table_errors"]) + len(result["priority_table_errors"])
        print(f"=== {path} ===  ({n} table errors found)")
        for e in result["category_table_errors"]:
            print(f"  [category] ad {e['ad_id']}: claimed={e['claimed_label']} real={e['real_label']}")
        for e in result["priority_table_errors"]:
            print(f"  [confidence] ad {e['ad_id']}: claimed={e['claimed_confidence']} real={e['real_confidence']} "
                  f"(matches ad(s) {e['matches_other_ad']})" if e['matches_other_ad'] else
                  f"  [confidence] ad {e['ad_id']}: claimed={e['claimed_confidence']} real={e['real_confidence']}")
        print()

    total = sum(len(r["category_table_errors"]) + len(r["priority_table_errors"]) for r in results)
    runs_with_errors = sum(1 for r in results if r["category_table_errors"] or r["priority_table_errors"])
    print(f"TOTAL: {total} structured-table errors across {len(results)} runs; "
          f"{runs_with_errors}/{len(results)} runs had at least one")

    with open("agent/results/conflation_check.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved to agent/results/conflation_check.json")


if __name__ == "__main__":
    main()
