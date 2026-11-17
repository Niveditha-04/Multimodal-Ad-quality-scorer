"""Traceable model_version string for the scores table.

The version string is a tag plus a short hash of the weight file's content,
so it always identifies the exact weights that produced a score. Retraining
classifier_head.pt changes the hash automatically and old scores stay
attributable to the weights that produced them. The tag names the classifier
family and appears in stored scores, so it is kept stable.
"""
import hashlib
from pathlib import Path

WEIGHTS_PATH = Path("model/classifier_head.pt")
VERSION_TAG = "joint_v2"


def get_model_version(weights_path: Path = WEIGHTS_PATH, version_tag: str = VERSION_TAG) -> str:
    """weights_path and version_tag default to the joint classifier.
    api/main.py passes its CLASSIFIER_CHECKPOINT and tag here when serving a
    compressed checkpoint, so model_version always tracks the weights in use.
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"{weights_path} not found -- train the classifier first")
    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()[:8]
    return f"{version_tag}_{digest}"


if __name__ == "__main__":
    print(get_model_version())
