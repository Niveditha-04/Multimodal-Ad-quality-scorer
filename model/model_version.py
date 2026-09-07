"""Traceable model_version string for the scores table.

Ties the version string to the actual weight file's content (via a short
hash) rather than a hand-maintained label, so it can't drift out of sync
with what's actually running -- if classifier_head.pt is retrained, the hash
changes automatically and old scores stay attributable to the exact weights
that produced them. The "joint_v2_postfix" tag documents which known fix
generation this is (post the misleading-label breed-name leak fix); a third
retrain would get "joint_v3" or similar plus its own new hash.
"""
import hashlib
from pathlib import Path

WEIGHTS_PATH = Path("model/classifier_head.pt")
VERSION_TAG = "joint_v2_postfix"


def get_model_version(weights_path: Path = WEIGHTS_PATH, version_tag: str = VERSION_TAG) -> str:
    """weights_path/version_tag default to the original joint classifier so
    every existing caller is unaffected. api/main.py passes its actual
    CLASSIFIER_CHECKPOINT/tag here when serving a compression-study
    checkpoint, so model_version keeps tracking the real weights in use --
    the same discipline db/README.md documents for the rule-baseline arm:
    a version string must be tied to actual weight content, never assumed.
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"{weights_path} not found -- train the classifier first")
    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()[:8]
    return f"{version_tag}_{digest}"


if __name__ == "__main__":
    print(get_model_version())
