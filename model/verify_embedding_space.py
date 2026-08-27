"""Verifies model/clip_features.py's embed_batch() -- the shared extraction
function used by extract_embeddings.py and the /score API -- pulls from
CLIP's contrastively-trained projected space, not the pre-projection
sub-model pooler output.

This distinction matters and a naive shape check can't catch it: on
clip-vit-base-patch32, the vision tower's raw hidden size is 768 (would be an
obvious bug if it leaked through -- wrong shape entirely), but the text
tower's raw hidden size is also 512, identical to the projected dimension.
A text-side mixup would produce correctly-shaped, silently-wrong embeddings.

Cross-checks our extraction method against two independent references:
  - model(**inputs).image_embeds / .text_embeds: the documented, unambiguous
    full-forward-pass projected output. Should match our extraction exactly
    after L2-normalization (the forward pass normalizes; get_*_features does
    not).
  - model.vision_model(...)/model.text_model(...).pooler_output: the raw
    PRE-projection pooler, used as a negative control. Our extraction should
    be essentially uncorrelated with this, not close to it.
"""
import torch
from PIL import Image

from model.clip_features import load_clip_model, embed_batch

MODEL_NAME = "openai/clip-vit-base-patch32"


def main() -> None:
    # method under test: the actual shared extraction path used by both
    # extract_embeddings.py and (from Phase 3 on) the /score API
    model, processor, device = load_clip_model()
    img = Image.open("data/images/0000_samoyed_64.jpg").convert("RGB")
    text = "a photo of a dog"

    img_extracted_np, txt_extracted_np = embed_batch(model, processor, device, [img], [text])
    img_extracted = torch.from_numpy(img_extracted_np)
    txt_extracted = torch.from_numpy(txt_extracted_np)

    # independent references, computed separately so this isn't just
    # checking embed_batch against itself
    inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        # reference: unambiguous full-forward-pass projected+normalized output
        full_out = model(**inputs)
        img_reference = full_out.image_embeds.cpu()
        txt_reference = full_out.text_embeds.cpu()

        # negative control: raw pre-projection pooler
        text_raw_pooler = model.text_model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
        ).pooler_output.cpu()

    img_norm = img_extracted / img_extracted.norm(dim=-1, keepdim=True)
    txt_norm = txt_extracted / txt_extracted.norm(dim=-1, keepdim=True)

    img_match = (img_norm - img_reference).abs().max().item()
    txt_match = (txt_norm - txt_reference).abs().max().item()
    cos_control = torch.nn.functional.cosine_similarity(txt_extracted, text_raw_pooler).item()

    print(f"image: max abs diff vs reference (after L2-norm) = {img_match:.2e}  (expect ~0)")
    print(f"text:  max abs diff vs reference (after L2-norm) = {txt_match:.2e}  (expect ~0)")
    print(f"text:  cosine sim vs raw pre-projection pooler   = {cos_control:.4f}  (expect near 0, NOT near 1)")

    assert img_match < 1e-4, "image embeddings do NOT match the documented projected space"
    assert txt_match < 1e-4, "text embeddings do NOT match the documented projected space"
    assert abs(cos_control) < 0.3, "text embeddings look suspiciously close to the raw pre-projection pooler"

    print("\nPASS: extraction confirmed to use CLIP's projected, contrastively-trained embedding space for both modalities.")


if __name__ == "__main__":
    main()
