"""Retrieval for the explanation layer: given a flagged ad's text, pull the
most relevant policy rule(s) and similar past flagged examples.

Policy rule retrieval is hybrid, not pure semantic search. Tested pure
semantic search first (see rag/verify_retrieval.py) and found it unreliable
at this corpus size (7 short rule documents): a garbled-spam query ranked
"low_quality_creative" 5th of 7 even though that rule's text literally
contains the phrase "keyword-stuffed spam" -- MiniLM's sentence embeddings
weighted overall register (formal explanatory prose vs. a shouty fragment)
over the keyword match. Since the classifier's predicted_label is already
known at /score time, "misleading" and "low_quality" map 1:1 onto a specific
rule -- that rule is guaranteed-included rather than left to embedding luck.
"policy_violation" has no 1:1 mapping (it covers 4 different sub-rules --
health claims, superlatives, fake urgency, prohibited language) and stays on
pure semantic search, which tested well for exactly this disambiguation.
"""
import chromadb

DB_PATH = "rag/chroma_db"
COLLECTION_NAME = "ad_policy_rag"

LABEL_TO_POLICY_CATEGORY = {
    "misleading": "mismatched_creative",
    "low_quality": "low_quality_creative",
}

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
        _collection = _client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve_context(ad_text: str, predicted_label: str | None = None, n_policy: int = 2, n_examples: int = 2) -> dict:
    collection = get_collection()

    policy_rules = []
    guaranteed_category = LABEL_TO_POLICY_CATEGORY.get(predicted_label)
    if guaranteed_category:
        guaranteed = collection.get(ids=[f"policy_{guaranteed_category}"])
        policy_rules.append({
            "category_name": guaranteed["metadatas"][0]["category_name"],
            "rule_description": guaranteed["documents"][0],
            "distance": None,  # guaranteed by label mapping, not ranked by similarity
        })

    remaining_slots = n_policy - len(policy_rules)
    if remaining_slots > 0:
        policy_results = collection.query(
            query_texts=[ad_text], n_results=remaining_slots + len(policy_rules),
            where={"type": "policy_rule"},
        )
        for doc, meta, dist in zip(
            policy_results["documents"][0], policy_results["metadatas"][0], policy_results["distances"][0],
        ):
            if meta["category_name"] == guaranteed_category:
                continue  # already included above
            if len(policy_rules) >= n_policy:
                break
            policy_rules.append({"category_name": meta["category_name"], "rule_description": doc, "distance": dist})

    example_results = collection.query(
        query_texts=[ad_text], n_results=n_examples,
        where={"type": "flagged_example"},
    )
    examples = [
        {"ad_text": doc, "ground_truth_label": meta["ground_truth_label"], "distance": dist}
        for doc, meta, dist in zip(
            example_results["documents"][0], example_results["metadatas"][0], example_results["distances"][0],
        )
    ]
    return {"policy_rules": policy_rules, "examples": examples}
