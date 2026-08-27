"""Loads data/dataset.csv into the `ads` table and seeds `policy_categories`.

Idempotent-ish: refuses to run if the ads table already has rows, so re-running
doesn't silently duplicate the dataset. Delete db/ads.db to start over.
"""
import sys

import pandas as pd

from db.models import Ad, PolicyCategory
from db.policy_categories import POLICY_CATEGORIES
from db.session import SessionLocal, init_db


def main() -> None:
    init_db()
    session = SessionLocal()

    existing_ads = session.query(Ad).count()
    if existing_ads > 0:
        print(f"ads table already has {existing_ads} rows -- refusing to reload. "
              f"Delete db/ads.db first if you want a clean reload.")
        sys.exit(1)

    df = pd.read_csv("data/dataset.csv")
    for _, row in df.iterrows():
        session.add(Ad(
            image_path=row["image_path"],
            ad_text=row["ad_text"],
            ground_truth_label=row["ground_truth_label"],
        ))
    session.commit()
    print(f"loaded {len(df)} ads into db")

    existing_cats = session.query(PolicyCategory).count()
    if existing_cats == 0:
        for cat in POLICY_CATEGORIES:
            session.add(PolicyCategory(
                category_name=cat["category_name"],
                rule_description=cat["rule_description"],
            ))
        session.commit()
        print(f"loaded {len(POLICY_CATEGORIES)} policy categories into db")
    else:
        print(f"policy_categories already has {existing_cats} rows -- skipped")

    session.close()


if __name__ == "__main__":
    main()
