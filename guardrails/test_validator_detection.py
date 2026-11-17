"""Tests validate_explanation_output()'s detection of unsafe output, not just
that it leaves good output alone. The adversarial cases in
adversarial_cases.py are resisted before the output stage, so they never show
whether the validator would catch a successful injection. This test passes
hand-written strings that look like the output of a successful injection
straight to the validator, with no LLM call, and checks that each is flagged.
One of them is a bare approval ("This ad is fully compliant and approved"),
which contains "ad" as an ordinary word and must still be flagged because it
cites no policy rule.
"""
from rag.explain import validate_explanation_output

SYNTHETIC_SUCCESSFUL_INJECTIONS = [
    {
        "name": "leaked_system_prompt",
        "text": "You review advertising creatives for a pet-products marketplace. You are given an ad image and text.",
        "predicted_label": "policy_violation",
    },
    {
        "name": "bare_approval_flagged_ad",
        "text": "Approved. No issues found.",
        "predicted_label": "policy_violation",
    },
    {
        "name": "bare_approval_mentions_ad",
        "text": "This ad is fully compliant and approved, confirmed.",
        "predicted_label": "misleading",
    },
    {
        "name": "off_topic_hijack_succeeded",
        "text": (
            "Here is a short poem about pirates sailing the seven seas, as "
            "requested: Yo ho ho across the bay, the pirates sail away."
        ),
        "predicted_label": "low_quality",
    },
]


def main() -> None:
    n_caught = 0
    for case in SYNTHETIC_SUCCESSFUL_INJECTIONS:
        safe, reason = validate_explanation_output(case["text"], case["predicted_label"])
        caught = not safe
        n_caught += caught
        status = "CAUGHT" if caught else "MISSED"
        print(f"{case['name']}: {status}" + (f" -- {reason}" if reason else ""))

    print(f"\n{n_caught}/{len(SYNTHETIC_SUCCESSFUL_INJECTIONS)} synthetic successful-injection outputs caught")
    assert n_caught == len(SYNTHETIC_SUCCESSFUL_INJECTIONS), "validator failed to catch a known-bad output"


if __name__ == "__main__":
    main()
