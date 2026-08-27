"""Rule-based baseline classifier (Arm A for the Phase 5 comparison).

Three independently calibrated heuristics, each tuned ONLY on data/split.json's
TRAINING ad ids (never the test set -- same discipline as the ML classifier,
see model/compute_confidence_intervals.py and db/README.md). Calibration
evidence and thresholds chosen are documented in each function's docstring.

Deliberately does NOT attempt to detect "misleading" (image-text species
mismatch). This isn't an oversight -- a text/basic-image rule system has no
way to determine what species is actually pictured without some form of
image understanding model, which would defeat the point of a "simple
baseline." Arm A's recall on misleading is expected to be near zero, and
that gap is exactly what motivates the multimodal classifier's existence.
Precedence when a text/image could trigger multiple rules: check
policy_violation first (most specific, keyword-grounded), then low_quality,
else approved.
"""
import re

import numpy as np
from PIL import Image, ImageFilter

# calibrated on train split: 100% recall, 0% false positives on the other
# 3 classes (see ab_test/calibration_log.md)
BLOCKLIST_PATTERNS = [
    r"\bcure[sd]?\b", r"\bguarantee[sd]?\b", r"#1\b", r"clinically proven",
    r"\bonly \d+ left\b", r"sale ends", r"last chance", r"before it'?s gone",
    r"\beliminate[s]?\b", r"\breverses?\b", r"stop your", r"suffering",
    r"nothing else.*compares", r"hands down", r"buy now before", r"comes close",
]
_BLOCKLIST_RE = [re.compile(p, re.IGNORECASE) for p in BLOCKLIST_PATTERNS]

# calibrated on train split: 0% false positives on the other 3 classes,
# ~45% recall on low_quality (catches the text-garbled half; structurally
# cannot catch the image-degraded half, which the blur heuristic below covers)
CAPS_RATIO_THRESHOLD = 0.5
EXCLAMATION_THRESHOLD = 3
MAX_NGRAM_CHECK = 4

# calibrated on train split: 400 gives 0% false positives on approved,
# catches 51/96 (53%) of low_quality -- right at the ~50% ceiling since only
# half of low_quality is image-degraded (see Phase 5 calibration run)
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
