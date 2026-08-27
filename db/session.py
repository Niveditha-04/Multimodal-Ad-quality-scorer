"""Engine/session setup. SQLite for local dev; swapping to Postgres is just
changing DATABASE_URL since everything goes through SQLAlchemy's ORM.

load_dotenv() is called HERE, not just in rag/explain.py, and deliberately
before DATABASE_URL is read. api/main.py imports this module at the top
(module load time), while rag/explain.py -- previously the only place
calling load_dotenv() -- is only imported lazily inside the /score handler.
That meant DATABASE_URL set in .env was silently ignored (engine already
built against the SQLite default before .env was ever loaded) unless it
happened to also be exported in the shell. Verified this was a real bug,
not a hypothetical: with DATABASE_URL only in .env, engine.url resolved to
sqlite:///./db/ads.db even after adding a postgresql:// line to .env.
"""
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./db/ads.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
