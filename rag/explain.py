"""Placeholder for Phase 4 (RAG + LLM explanation layer).

api/main.py imports generate_explanation() from here so /score is wired up
end-to-end now. This stub makes that honest about its own state rather than
silently returning something that looks like a real explanation -- it will
be replaced with actual retrieval + Claude API calls in Phase 4.
"""


def generate_explanation(ad_text: str, predicted_label: str) -> str:
    return (
        f"[Phase 4 not yet built] This ad was flagged as '{predicted_label}'. "
        f"A grounded explanation citing the relevant policy rule and similar "
        f"past examples will be generated here once the RAG + LLM layer is implemented."
    )
