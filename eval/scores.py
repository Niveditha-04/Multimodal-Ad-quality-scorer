"""Per-example scores against eval/rubric.md, applied by direct human/agent
review of each of the 32 generated explanations against golden-set ground
truth -- not delegated to another LLM call, which would risk the same
correlated-error problem as using a model to grade itself. Scored after
generation, against the rubric as already locked in eval/rubric.md; the
rubric was not adjusted based on what these explanations looked like.

Each entry: (groundedness, rule_accuracy, hallucination), each 0/1/2, plus a
one-line note on the reasoning for any score below 2.
"""

SCORES = {
    # --- misclassification cases (ground truth != model prediction) ---
    6: (2, 0, 0, "Accurately describes image (Birman cat outdoors); genuine_rule=none so any asserted rule is auto-0; fabricates 'product must be visible in frame' requirement, the v1-known hallucination, reproduced identically."),
    97: (2, 0, 0, "Accurately describes image (Keeshond puppy), even hedges 'breed shown may technically align'; genuine_rule=none -> auto-0; same fabricated product-visibility requirement."),
    105: (2, 0, 0, "Accurately describes image (Ragdoll cat); genuine_rule=none -> auto-0; same fabricated product-visibility requirement, third occurrence."),
    146: (2, 1, 1, "Correctly notes species matches (no image error); genuine_rule=unsubstantiated_health_claims but cites absolute_superlative_claims -- ad genuinely contains both violation types textually, so not fabricated, just not the pre-committed primary rule; also tacks on the same product-visibility fabrication as secondary, non-central support."),

    # --- policy_violation true positives ---
    140: (1, 2, 2, "No image description at all (rubric: no-description = 1, not 2); correct rule, accurate quote, no fabrication."),
    144: (0, 1, 0, "Misidentifies breed as 'Goldendoodle' when ground truth is wheaten terrier (specific, checkable, wrong); correct primary rule (fake_urgency_or_scarcity) but adds a fabricated second violation (mismatched_creative) asserted as co-equal reasoning."),
    145: (1, 2, 2, "No image description; correct rule, accurate, no fabrication."),
    157: (2, 1, 0, "Accurate image description; genuine_rule=prohibited_language but cites absolute_superlative_claims (text does contain overlapping absolute-claim language, so plausible not fabricated); adds fabricated mismatched_creative as a co-equal second violation."),
    161: (2, 2, 1, "Correct rule (unsubstantiated_health_claims) stated first and accurately; image description plausible (not independently re-verified at scoring time); fabricated product-visibility point added but framed as subordinate/compounding, not an independent violation."),
    168: (1, 2, 2, "No image description; correct rule, accurate, no fabrication."),
    170: (1, 2, 2, "No image description; correct rule, accurate, no fabrication."),
    178: (1, 2, 2, "No image description; correct rule, accurate, no fabrication."),
    185: (0, 1, 0, "Claims pictured dog 'is not identifiable as an American Pit Bull Terrier' -- ground truth confirms it genuinely is (visually verified); genuine_rule=prohibited_language but cites absolute_superlative_claims; fabricates a breed-mismatch violation asserted as co-equal reasoning."),
    211: (2, 2, 0, "Accurate breed ID (Boxer, matches ground truth); correct rule stated first and precisely; adds fabricated mismatched_creative as an explicit co-equal second violation ('flagged... on two counts')."),

    # --- misleading true positives (genuine species mismatch -- the case
    # where the guaranteed-rule retrieval is actually correct) ---
    243: (2, 2, 2, "Accurate breed ID (Persian); correct rule, genuinely and accurately applied to a real mismatch; no fabrication."),
    249: (2, 2, 2, "Accurate breed ID (Cocker Spaniel); correct rule, real mismatch, no fabrication."),
    277: (2, 2, 2, "Accurate breed ID (Siamese); correct rule, real mismatch, no fabrication."),
    300: (2, 2, 2, "Accurate description (black cat on plaid blanket, matches visual confirmation); correct rule, real mismatch, no fabrication."),
    320: (2, 2, 2, "Accurate breed ID (Leonberger); correct rule, real mismatch, no fabrication."),
    321: (2, 2, 2, "Accurate breed ID (Sphynx); correct rule, real mismatch, no fabrication."),
    333: (2, 2, 2, "Accurate breed ID (Japanese Chin); correct rule, real mismatch, no fabrication."),
    347: (2, 2, 2, "Accurate, appropriately hedged breed ID ('Miniature Pinscher or similar'); correct rule, real mismatch, no fabrication."),
    359: (2, 2, 2, "Accurate breed ID (Shiba Inu); correct rule, real mismatch, no fabrication."),

    # --- low_quality true positives ---
    367: (2, 2, 2, "Accurately describes severe blur, appropriately hedged ('appears to show'); correctly identifies text as coherent (correct subtype ID); correct rule, no fabrication."),
    388: (2, 2, 2, "Correctly identifies text-spam as the issue, correctly notes image is fine and relevant (cat, matches ground truth); correct rule, no fabrication."),
    396: (1, 2, 2, "No image description; correct rule, accurate quote, no fabrication."),
    412: (2, 1, 0, "Accurate breed ID (Shiba Inu) and plausible specific image detail (watermark text); but incorrectly asserts the (ground-truth-normal) image is itself 'generic' and contributes to the low_quality flag ('mismatched generic image constitute a low-quality creative') -- same product/image-relevance fabrication pattern leaking into a low_quality explanation, presented as central supporting reasoning alongside the correct text-spam finding."),
    418: (2, 2, 2, "Correctly identifies text-spam, correctly notes image (Siamese cat) is relevant and fine; correct rule, no fabrication."),
    429: (2, 2, 2, "Accurately describes severe blur (matches direct visual confirmation of this exact image); correct rule, no fabrication."),
    436: (2, 2, 2, "Accurate breed ID (Boxer puppy); correctly identifies text-spam as sole issue; correct rule, no fabrication."),
    450: (1, 2, 2, "No image description; correct rule, accurate quote, no fabrication."),
    461: (2, 2, 2, "Appropriately hedged blur description ('appears to show... too blurry to identify'); correct rule, no fabrication."),
}
