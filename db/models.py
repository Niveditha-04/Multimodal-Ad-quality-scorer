"""SQLAlchemy schema for the ad quality scorer.

Three tables:
  - ads: one row per ad creative (image + copy), with the ground-truth label
    used for training/eval. Ground truth is only populated for the labeled
    dataset; ads scored live through the API leave it null.
  - scores: one row per scoring event. An ad can be scored more than once
    (different model versions), so this is a separate table rather than
    columns on `ads`.
  - policy_categories: static reference table of the policy rules the
    RAG layer retrieves against.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Ad(Base):
    __tablename__ = "ads"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String, nullable=False)
    ad_text = Column(Text, nullable=False)
    ground_truth_label = Column(String, nullable=True)  # null for live-scored ads with no known label
    created_at = Column(DateTime, default=utcnow, nullable=False)

    scores = relationship("Score", back_populates="ad", cascade="all, delete-orphan")


class Score(Base):
    __tablename__ = "scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ad_id = Column(Integer, ForeignKey("ads.id"), nullable=False)
    predicted_label = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    model_version = Column(String, nullable=False)
    scored_at = Column(DateTime, default=utcnow, nullable=False)

    ad = relationship("Ad", back_populates="scores")


class PolicyCategory(Base):
    __tablename__ = "policy_categories"

    category_id = Column(Integer, primary_key=True, autoincrement=True)
    category_name = Column(String, nullable=False, unique=True)
    rule_description = Column(Text, nullable=False)
