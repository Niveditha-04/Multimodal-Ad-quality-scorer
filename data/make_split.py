"""Creates the single train/test split every downstream model must reuse:
text-only ablation, image-only ablation, joint classifier, and the rule-based
baseline in Phase 5. If each model split independently, comparisons between
them would be meaningless -- one could look better purely from an easier
random test set, not from being an actually better model.

Split is grouped by source_image (the underlying photo, before our per-row
prefix), not by row, so that if a photo were ever reused across rows, all of
its rows would land on the same side of the split. check_leakage.py verified
there's currently no reuse (480 rows -> 480 unique source images by content
hash), so this reduces to a plain stratified split in practice -- but the
code path is group-aware regardless, since that's what's actually correct.

Requires the ads table to exist (db/load_dataset.py already run) since we
split on ad_id, the id every other table joins against.
"""
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SPLIT_PATH = Path("data/split.json")
TEST_SIZE = 0.2
RANDOM_SEED = 42


def source_filename(image_path: str) -> str:
    basename = Path(image_path).name
    return re.sub(r"^\d{4}_", "", basename)


def main() -> None:
    import sqlite3
    conn = sqlite3.connect("db/ads.db")
    ads = pd.read_sql("SELECT id, image_path, ground_truth_label FROM ads ORDER BY id", conn)
    conn.close()

    ads["source_image"] = ads["image_path"].apply(source_filename)
    n_groups = ads["source_image"].nunique()
    print(f"{len(ads)} ads, {n_groups} unique source-image groups")

    # group-aware stratified-ish split: GroupShuffleSplit respects groups but
    # doesn't stratify directly, so we split each label's groups separately
    # and concatenate -- this keeps both the grouping guarantee AND label
    # balance across train/test.
    train_ids, test_ids = [], []
    for label, sub in ads.groupby("ground_truth_label"):
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        train_idx, test_idx = next(gss.split(sub, groups=sub["source_image"]))
        train_ids.extend(sub.iloc[train_idx]["id"].tolist())
        test_ids.extend(sub.iloc[test_idx]["id"].tolist())

    train_ids.sort()
    test_ids.sort()

    assert set(train_ids).isdisjoint(set(test_ids)), "train/test overlap!"
    assert len(train_ids) + len(test_ids) == len(ads), "lost or duplicated ad ids"

    # verify no source_image group is split across train/test
    id_to_group = dict(zip(ads["id"], ads["source_image"]))
    train_groups = {id_to_group[i] for i in train_ids}
    test_groups = {id_to_group[i] for i in test_ids}
    leaked_groups = train_groups & test_groups
    assert not leaked_groups, f"source-image group leaked across split: {leaked_groups}"

    split = {
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "grouped_by": "source_image",
        "train_ad_ids": train_ids,
        "test_ad_ids": test_ids,
    }
    SPLIT_PATH.write_text(json.dumps(split, indent=2))
    print(f"wrote {SPLIT_PATH}: {len(train_ids)} train, {len(test_ids)} test")

    train_df = ads[ads["id"].isin(train_ids)]
    test_df = ads[ads["id"].isin(test_ids)]
    print("\ntrain label counts:")
    print(train_df["ground_truth_label"].value_counts())
    print("\ntest label counts:")
    print(test_df["ground_truth_label"].value_counts())


if __name__ == "__main__":
    main()
