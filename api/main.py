"""FastAPI service exposing the trained ad-quality classifier.

POST /score  - score one (image, text) ad creative, persist it + the
               prediction, and (if flagged) attach an LLM explanation
               grounded in retrieved policy text (Phase 4).
GET  /stats  - runs the "violation rate by predicted category" query from
               db/queries.sql against whatever has actually been scored.

Loads the CLIP backbone and classifier head once at startup, not per
request -- CLIP inference on CPU/MPS per request is the expensive part, and
reloading it on every call would make this unusably slow. Both come from
the shared model/clip_features.py and model/classifier_architecture.py
modules used at training time, so a live request produces embeddings in
exactly the same space the classifier was trained on.
"""
import io
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from sqlalchemy import text

from db.models import Ad, Score
from db.session import SessionLocal, init_db
from model.clip_features import load_clip_model, embed_batch, build_joint_features
from model.classifier_architecture import LABELS, MLPHead, HIDDEN_DIM, DROPOUT
from model.model_version import get_model_version

ml_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    model, processor, device = load_clip_model()
    ml_state["clip_model"] = model
    ml_state["clip_processor"] = processor
    ml_state["device"] = device

    classifier = MLPHead(input_dim=1024, hidden_dim=HIDDEN_DIM, n_classes=len(LABELS), dropout=DROPOUT)
    state_dict = torch.load("model/classifier_head.pt", map_location="cpu")
    classifier.load_state_dict(state_dict)
    classifier.eval()
    ml_state["classifier"] = classifier
    ml_state["model_version"] = get_model_version()

    print(f"loaded CLIP on {device}, classifier head, model_version={ml_state['model_version']}")
    yield
    ml_state.clear()


app = FastAPI(title="Ad Quality & Policy Violation Scorer", lifespan=lifespan)


class ScoreResponse(BaseModel):
    ad_id: int
    predicted_label: str
    confidence: float
    model_version: str
    explanation: str | None = None


@app.post("/score", response_model=ScoreResponse)
async def score_ad(image: UploadFile = File(...), text: str = Form(...)):
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    raw_bytes = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="uploaded file is not a readable image")

    image_embeds, text_embeds = embed_batch(
        ml_state["clip_model"], ml_state["clip_processor"], ml_state["device"],
        [pil_image], [text],
    )
    features = build_joint_features(image_embeds, text_embeds)

    with torch.no_grad():
        logits = ml_state["classifier"](torch.tensor(features, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])

    predicted_label = LABELS[pred_idx]

    # persist the ad + score; image bytes themselves are not stored on disk
    # here (no durable image store wired up for the API path) -- image_path
    # records that this came through /score rather than the labeled dataset
    session = SessionLocal()
    try:
        ad = Ad(
            image_path=f"<uploaded:{image.filename}>",
            ad_text=text,
            ground_truth_label=None,
        )
        session.add(ad)
        session.flush()  # populates ad.id without committing yet

        score = Score(
            ad_id=ad.id,
            predicted_label=predicted_label,
            confidence=confidence,
            model_version=ml_state["model_version"],
        )
        session.add(score)
        session.commit()
        ad_id = ad.id
    finally:
        session.close()

    explanation = None
    if predicted_label != "approved":
        from rag.explain import generate_explanation
        explanation = generate_explanation(ad_text=text, predicted_label=predicted_label, image=pil_image)

    return ScoreResponse(
        ad_id=ad_id,
        predicted_label=predicted_label,
        confidence=confidence,
        model_version=ml_state["model_version"],
        explanation=explanation,
    )


@app.get("/stats")
def get_stats(model_version: str | None = None):
    # scores accumulates rows from multiple model_versions (this classifier,
    # the rule-based baseline, past retrains) -- see db/README.md. Every
    # reference query in db/queries.sql filters by model_version for exactly
    # this reason; this endpoint defaults to the currently-loaded model so a
    # caller gets that model's own stats, not a blend across arms. Pass
    # ?model_version=... to inspect a different arm (e.g. the baseline).
    filter_version = model_version or ml_state["model_version"]

    session = SessionLocal()
    try:
        rows = session.execute(
            text(
                """
                SELECT predicted_label, COUNT(*) AS n_predictions
                FROM scores
                WHERE model_version = :model_version
                GROUP BY predicted_label
                ORDER BY n_predictions DESC
                """
            ),
            {"model_version": filter_version},
        ).fetchall()
        total = session.execute(
            text("SELECT COUNT(*) FROM scores WHERE model_version = :model_version"),
            {"model_version": filter_version},
        ).scalar()
    finally:
        session.close()

    if total == 0:
        return {"model_version": filter_version, "total_scored": 0, "by_predicted_label": []}

    return {
        "model_version": filter_version,
        "total_scored": total,
        "by_predicted_label": [
            {
                "predicted_label": r[0],
                "n_predictions": r[1],
                "pct_of_all_scores": round(100.0 * r[1] / total, 2),
            }
            for r in rows
        ],
    }
