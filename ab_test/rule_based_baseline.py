"""Rule-based baseline classifier (Arm A for the comparison).

Three heuristics, each calibrated only on the 384 training ad ids in
data/split.json (the test set is never used). The calibration results sit
next to each threshold below.

The baseline does not try to detect "misleading" (image-text species
mismatch). A text and basic-image rule system cannot tell what species a
photo shows without an image-understanding model, so its recall on that
class is expected to be near zero.
Precedence when several rules could fire: policy_violation first (most
specific, keyword-based), then low_quality, else approved.
"""
import re

import numpy as np
from PIL import Image, ImageFilter

# Blocklist, train split: 96/96 policy_violation caught, 0 false positives on
# the other three classes.
BLOCKLIST_PATTERNS = [
    r"\bcure[sd]?\b", r"\bguarantee[sd]?\b", r"#1\b", r"clinically proven",
    r"\bonly \d+ left\b", r"sale ends", r"last chance", r"before it'?s gone",
    r"\beliminate[s]?\b", r"\breverses?\b", r"stop your", r"suffering",
    r"nothing else.*compares", r"hands down", r"buy now before", r"comes close",
]
_BLOCKLIST_RE = [re.compile(p, re.IGNORECASE) for p in BLOCKLIST_PATTERNS]

# Spam text rule, train split: 43/96 (44.8%) low_quality caught, 0 false
# positives on the other three classes. It catches the text-garbled half of
# low_quality and cannot see the image-degraded half.
CAPS_RATIO_THRESHOLD = 0.5
EXCLAMATION_THRESHOLD = 3
MAX_NGRAM_CHECK = 4

# Blur rule, train split: a threshold of 400 flags 0/96 approved ads and
# catches 49/96 (51.0%) of low_quality. Higher thresholds catch more but flag
# approved ads too (500: 1/96, 600: 6/96, 800: 13/96, 1000: 26/96).
BLUR_EDGE_VARIANCE_THRESHOLD = 400


def has_violation_language(text: str) -> bool:
    return any(p.search(text) for p in _BLOCKLIST_RE)


def _has_repeated_ngram(text: str, max_n: int = MAX_NGRAM_CHECK) -> bool:
    words = text.lower().split()
    for n in range(1, max_n + 1):
        for i in range(len(words) - 2 * n + 1):
            if words[i:i + n] == words[i + n:i + 2 * n]:
                return True
    return False


def _caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 5:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def is_spammy_text(text: str) -> bool:
    return (
        _has_repeated_ngram(text)
        or _caps_ratio(text) > CAPS_RATIO_THRESHOLD
        or text.count("!") >= EXCLAMATION_THRESHOLD
    )


def is_blurry_image(image: Image.Image) -> bool:
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_variance = np.array(edges, dtype=np.float64).var()
    return edge_variance < BLUR_EDGE_VARIANCE_THRESHOLD


def predict(ad_text: str, image: Image.Image) -> str:
    if has_violation_language(ad_text):
        return "policy_violation"
    if is_spammy_text(ad_text) or is_blurry_image(image):
        return "low_quality"
    return "approved"
