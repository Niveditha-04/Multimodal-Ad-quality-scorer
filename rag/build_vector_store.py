"""Builds the RAG vector store: 7 policy rules + past flagged examples.

Flagged examples are pulled from the TRAINING split only (never test), same
reasoning as keeping the test set out of everything else in this project --
if the retrieval corpus contained test-set ads, a live /score call on a
test-set image+text pair could retrieve a near-identical past example and
make the explanation look suspiciously well-grounded for reasons that have
nothing to do with the retrieval actually being good.

Uses Chroma's default embedding function (all-MiniLM-L6-v2 via onnxruntime,
downloaded once) rather than CLIP's text encoder -- CLIP's text tower is
trained for cross-modal image-text alignment, not text-to-text semantic
retrieval, so it's the wrong tool for this specific job even though it's
already loaded elsewhere in this project.
"""
import json
import sqlite3

import chromadb

from db.policy_categories import POLICY_CATEGORIES

DB_PATH = "rag/chroma_db"
COLLECTION_NAME = "ad_policy_rag"


def main() -> None:
    client = chromadb.PersistentClient(path=DB_PATH)

    # start clean so re-runs don't duplicate documents
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids, documents, metadatas = [], [], []

    for cat in POLICY_CATEGORIES:
        ids.append(f"policy_{cat['category_name']}")
        documents.append(cat["rule_description"])
        metadatas.append({"type": "policy_rule", "category_name": cat["category_name"]})

    split = json.loads(open("data/split.json").read())
    train_ids = set(split["train_ad_ids"])

    conn = sqlite3.connect("db/ads.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT id, ad_text, ground_truth_label FROM ads "
        "WHERE ground_truth_label IS NOT NULL AND ground_truth_label != 'approved'"
    )
    rows = cur.fetchall()
    conn.close()

    n_examples = 0
    for ad_id, ad_text, label in rows:
        if ad_id not in train_ids:
            continue  # test-set ads never enter the retrieval corpus
        ids.append(f"example_{ad_id}")
        documents.append(ad_text)
        metadatas.append({"type": "flagged_example", "ground_truth_label": label, "ad_id": ad_id})
        n_examples += 1

    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"added {len(POLICY_CATEGORIES)} policy rules + {n_examples} flagged examples")
    print(f"collection count: {collection.count()}")


if __name__ == "__main__":
    main()
