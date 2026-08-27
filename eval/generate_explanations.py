"""Generates an explanation for every golden-set ad via the real
rag.explain.generate_explanation() pipeline -- same function /score and the
MCP tool call, not a reimplementation. Uses each ad's already-recorded
model_predicted_label (from the actual classifier run stored in the DB),
not a fresh classification -- this isolates what's being evaluated to the
explanation layer specifically, holding the classifier's decision fixed.
"""
import json

from PIL import Image

from rag.explain import generate_explanation


def main() -> None:
    golden_set = json.loads(open("eval/golden_set.json").read())

    results = []
    for i, entry in enumerate(golden_set, 1):
        image = Image.open(entry["image_path"]).convert("RGB")
        explanation = generate_explanation(
            ad_text=entry["ad_text"],
            predicted_label=entry["model_predicted_label"],
            image=image,
        )
        results.append({"ad_id": entry["ad_id"], "explanation": explanation})
        print(f"[{i}/{len(golden_set)}] ad_id={entry['ad_id']} ({entry['model_predicted_label']}) done")

    with open("eval/generated_explanations.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {len(results)} explanations to eval/generated_explanations.json")


if __name__ == "__main__":
    main()
