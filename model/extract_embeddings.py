"""Extracts frozen CLIP image and text embeddings for every ad in the DB.

Bulk driver around model/clip_features.py -- the actual extraction logic
lives there so the /score API (Phase 3) can call the identical function on
one (image, text) pair at a time, instead of reimplementing it and risking
train-serve skew.

Saved once here so every downstream script (ablations, joint classifier,
Phase 5 baseline comparison) reads the same embeddings instead of
re-running CLIP inference repeatedly.
"""
import sqlite3

import numpy as np
import pandas as pd
from PIL import Image

from model.clip_features import embed_batch, load_clip_model

OUT_PATH = "model/embeddings.npz"
BATCH_SIZE = 16


def main() -> None:
    model, processor, device = load_clip_model()
    print(f"using device: {device}")

    conn = sqlite3.connect("db/ads.db")
    ads = pd.read_sql("SELECT id, image_path, ad_text, ground_truth_label FROM ads ORDER BY id", conn)
    conn.close()
    print(f"embedding {len(ads)} ads")

    all_image_embeds = []
    all_text_embeds = []

    for start in range(0, len(ads), BATCH_SIZE):
        batch = ads.iloc[start:start + BATCH_SIZE]
        images = [Image.open(p).convert("RGB") for p in batch["image_path"]]
        texts = batch["ad_text"].tolist()

        image_embeds, text_embeds = embed_batch(model, processor, device, images, texts)
        all_image_embeds.append(image_embeds)
        all_text_embeds.append(text_embeds)

        if start % (BATCH_SIZE * 5) == 0:
            print(f"  {start}/{len(ads)}")

    image_embeds = np.concatenate(all_image_embeds, axis=0)
    text_embeds = np.concatenate(all_text_embeds, axis=0)

    print(f"image_embeds shape: {image_embeds.shape}")
    print(f"text_embeds shape: {text_embeds.shape}")
    assert image_embeds.shape[0] == len(ads)
    assert text_embeds.shape[0] == len(ads)
    assert not np.isnan(image_embeds).any(), "NaNs in image embeddings"
    assert not np.isnan(text_embeds).any(), "NaNs in text embeddings"

    np.savez(
        OUT_PATH,
        ad_ids=ads["id"].values,
        image_embeds=image_embeds,
        text_embeds=text_embeds,
        labels=ads["ground_truth_label"].values,
    )
    print(f"saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
