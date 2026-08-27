"""Single source of truth for turning an (image, text) pair into the feature
vector the classifier head expects. Imported by extract_embeddings.py (bulk,
at training time) and by the /score API (one pair at a time, at serving
time) -- there is exactly one code path from raw input to model input, so
the API can't drift into a different embedding space than what the
classifier was trained on (train-serve skew).

CLIP backbone is frozen everywhere it's used: eval mode, no grad, weights
never updated.
"""
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_clip_model(device: torch.device | None = None) -> tuple[CLIPModel, CLIPProcessor, torch.device]:
    device = device or get_device()
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, processor, device


def embed_batch(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    images: Sequence[Image.Image],
    texts: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (image_embeds, text_embeds) in CLIP's projected space, raw
    (unnormalized) norm -- same convention as what's saved to embeddings.npz.
    Callers that need the classifier's actual input space must L2-normalize
    via l2_normalize() below; kept as a separate step so the raw projected
    vectors stay available for other uses (e.g. the cosine-similarity sanity
    checks in Phase 2).

    See model/verify_embedding_space.py for the empirical proof that
    .pooler_output on the RETURNED object from get_image_features/
    get_text_features holds the post-projection embedding, not the raw
    pre-projection sub-model pooler -- this is not obvious from the shape
    alone (see that script's docstring).
    """
    inputs = processor(
        text=list(texts), images=list(images), return_tensors="pt",
        padding=True, truncation=True,
    ).to(device)

    with torch.no_grad():
        image_out = model.get_image_features(pixel_values=inputs["pixel_values"])
        text_out = model.get_text_features(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
        )

    image_embeds = image_out.pooler_output.cpu().numpy()
    text_embeds = text_out.pooler_output.cpu().numpy()
    return image_embeds, text_embeds


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms


def build_joint_features(image_embeds: np.ndarray, text_embeds: np.ndarray) -> np.ndarray:
    """L2-normalizes each modality independently, then concatenates
    [image, text] in that order -- must match train_classifier.py exactly,
    since that order is baked into the trained weight matrix's input layout.
    """
    return np.concatenate([l2_normalize(image_embeds), l2_normalize(text_embeds)], axis=1)
