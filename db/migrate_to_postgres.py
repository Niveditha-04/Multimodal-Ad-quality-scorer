"""One-time data migration: copies every row from the existing SQLite
db/ads.db into Postgres (schema must already exist -- run db.session.init_db()
against DATABASE_URL pointed at Postgres first).

Preserves primary keys exactly, since data/split.json and other files
reference specific ad_ids. After inserting explicit IDs, Postgres's
auto-increment sequences must be reset to continue past the highest
migrated ID, or the next INSERT without an explicit id would collide.
"""
import os
import sqlite3

from sqlalchemy import create_engine, text

SQLITE_PATH = "db/ads.db"


def get_postgres_engine():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url or not database_url.startswith("postgresql"):
        raise RuntimeError(
            "DATABASE_URL must be set to a postgresql:// URL for this migration "
            f"(got: {database_url!r})"
        )
    return create_engine(database_url)


def main() -> None:
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_conn.row_factory = sqlite3.Row
    pg_engine = get_postgres_engine()

    with pg_engine.begin() as pg_conn:
        existing = pg_conn.execute(text("SELECT COUNT(*) FROM ads")).scalar()
        if existing > 0:
            raise RuntimeError(
                f"Postgres 'ads' table already has {existing} rows -- refusing to "
                f"migrate into a non-empty table. Drop and recreate the schema first."
            )

        for table, pk_col in [
            ("policy_categories", "category_id"),
            ("ads", "id"),
            ("scores", "id"),
        ]:
            rows = sqlite_conn.execute(f"SELECT * FROM {table}").fetchall()
            cols = rows[0].keys() if rows else []
            for row in rows:
                col_list = ", ".join(cols)
                placeholders = ", ".join(f":{c}" for c in cols)
                pg_conn.execute(
                    text(f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"),
                    dict(row),
                )
            print(f"migrated {len(rows)} rows into {table}")

            # reset the sequence to continue past the highest migrated id
            max_id = max((row[pk_col] for row in rows), default=0)
            seq_name = f"{table}_{pk_col}_seq"
            pg_conn.execute(text(f"SELECT setval(:seq, :val)"), {"seq": seq_name, "val": max_id})
            print(f"  reset sequence {seq_name} to {max_id}")

    sqlite_conn.close()
    print("migration complete")


if __name__ == "__main__":
    main()
