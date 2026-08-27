"""Engine/session setup. SQLite for local dev; swapping to Postgres is just
changing DATABASE_URL since everything goes through SQLAlchemy's ORM.
"""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./db/ads.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
