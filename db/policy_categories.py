"""Static policy rule set used to seed the `policy_categories` table and the
RAG vector store. These are written to look like a plausible internal ad-review
policy for a pet-products marketplace, not lifted from any real platform's
actual policy text.
"""

POLICY_CATEGORIES = [
    {
        "category_name": "unsubstantiated_health_claims",
        "rule_description": (
            "Ad copy must not claim a product cures, treats, prevents, or "
            "reverses a medical condition (e.g. arthritis, cancer, anxiety) "
            "unless the claim is substantiated by a cited clinical source. "
            "Vague wellness language ('supports joint health') is fine; "
            "absolute cure/guarantee language ('eliminates pain in days', "
            "'guaranteed to cure') is not."
        ),
    },
    {
        "category_name": "mismatched_creative",
        "rule_description": (
            "The product or animal shown in the image must match what the ad "
            "copy describes. An ad for a small-breed harness must not show a "
            "large-breed dog; an ad for cat litter must not show a dog. "
            "Mismatches between the pictured animal/product and the text "
            "description mislead the buyer about what they're purchasing."
        ),
    },
    {
        "category_name": "absolute_superlative_claims",
        "rule_description": (
            "Claims like 'the #1 vet-recommended product' or 'clinically "
            "proven best on the market' require a named, checkable source. "
            "Unqualified superlatives without substantiation are not allowed."
        ),
    },
    {
        "category_name": "fake_urgency_or_scarcity",
        "rule_description": (
            "Ad copy must not use fabricated urgency ('only 2 left', 'sale "
            "ends tonight') when no such time or inventory limit actually "
            "exists. This is a manipulative-pattern violation, independent "
            "of whether the product itself is accurately described."
        ),
    },
    {
        "category_name": "prohibited_language",
        "rule_description": (
            "Ad copy must not contain profanity, discriminatory language, or "
            "language designed to shock rather than inform (e.g. fear-based "
            "claims about a pet's safety to drive urgency)."
        ),
    },
    {
        "category_name": "low_quality_creative",
        "rule_description": (
            "This is a creative-quality standard, not a policy violation: "
            "images must be in focus and reasonably high resolution, and ad "
            "copy must be a coherent sentence describing the product, not "
            "keyword-stuffed spam or a fragment with no product information. "
            "Low-quality creatives are held for review separately from "
            "policy violations because the underlying claim may be truthful "
            "even though the execution is poor."
        ),
    },
    {
        "category_name": "accurate_compliant_ad",
        "rule_description": (
            "Baseline for approval: ad copy accurately describes the "
            "pictured product/animal, makes no unsubstantiated claims, uses "
            "no manipulative urgency, and the creative is in focus and "
            "legible. Ads meeting all of these are approved."
        ),
    },
]
