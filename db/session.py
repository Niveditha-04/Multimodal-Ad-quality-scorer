"""Engine/session setup. SQLite for local dev; swapping to Postgres is just
changing DATABASE_URL since everything goes through SQLAlchemy's ORM.

load_dotenv() runs here, before DATABASE_URL is read, so a value set only in
.env takes effect. api/main.py imports this module at startup, while
rag/explain.py (the other caller of load_dotenv) is imported lazily inside
the /score handler, after the engine has already been created.
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
