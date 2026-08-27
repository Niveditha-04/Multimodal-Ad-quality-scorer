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


def get_model_version() -> str:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"{WEIGHTS_PATH} not found -- train the classifier first")
    digest = hashlib.sha256(WEIGHTS_PATH.read_bytes()).hexdigest()[:8]
    return f"{VERSION_TAG}_{digest}"


if __name__ == "__main__":
    print(get_model_version())
