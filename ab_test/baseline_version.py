"""Traceable model_version string for the rule-based baseline, mirroring
model/model_version.py's approach for the joint classifier. The baseline's
thresholds and blocklist live in ab_test/rule_based_baseline.py. Hashing that
source file changes the version string whenever they change, so old and new
baseline runs never share a version string in the scores table.
"""
import hashlib
from pathlib import Path

SOURCE_PATH = Path("ab_test/rule_based_baseline.py")
VERSION_TAG = "rule_baseline_v1"


def get_baseline_version() -> str:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"{SOURCE_PATH} not found")
    digest = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()[:8]
    return f"{VERSION_TAG}_{digest}"


if __name__ == "__main__":
    print(get_baseline_version())
