"""Checks whether any two of the 480 rows share the same underlying source
photo. build_dataset.py samples from a shuffled list of unique files without
replacement, so by construction no source file should be used twice -- but
that's a claim about the generator, not evidence about the actual data.
This checks the real output: (a) same source filename appearing twice, and
(b) same file content (md5) appearing twice, in case two differently-named
files in the raw dataset are byte-identical.
"""
import hashlib
import re
from collections import Counter
from pathlib import Path

import pandas as pd

RAW_IMAGES_DIR = Path("data/raw/oxford-iiit-pet/images")


def source_filename(image_path: str) -> str:
    # our generated names are "<4-digit-index>_<original-filename>"
    basename = Path(image_path).name
    return re.sub(r"^\d{4}_", "", basename)


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    df = pd.read_csv("data/dataset.csv")
    df["source_image"] = df["image_path"].apply(source_filename)

    name_counts = Counter(df["source_image"])
    dup_names = {k: v for k, v in name_counts.items() if v > 1}
    print(f"rows: {len(df)}, unique source filenames: {df['source_image'].nunique()}")
    print(f"filenames used more than once: {len(dup_names)}")
    if dup_names:
        print(dup_names)

    hashes = df["source_image"].apply(lambda n: md5_of(RAW_IMAGES_DIR / n))
    hash_counts = Counter(hashes)
    dup_hashes = {k: v for k, v in hash_counts.items() if v > 1}
    print(f"unique content hashes: {hashes.nunique()}")
    print(f"content hashes used more than once: {len(dup_hashes)}")
    if dup_hashes:
        for h, cnt in dup_hashes.items():
            matches = df[hashes == h][["image_path", "ground_truth_label", "source_image"]]
            print(f"hash {h} appears {cnt} times:")
            print(matches.to_string(index=False))

    df[["image_path", "source_image"]].to_csv("data/source_image_map.csv", index=False)
    print("wrote data/source_image_map.csv")


if __name__ == "__main__":
    main()
