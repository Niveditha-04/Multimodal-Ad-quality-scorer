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

Which classifier checkpoint gets served is configurable via env vars, so
this same API can serve either the original trained head (the default,
unchanged behavior) or one of the pruned/quantized checkpoints from the
compression/ study:

  CLASSIFIER_CHECKPOINT  - path to the checkpoint file
                           (default: model/classifier_head.pt)
  CLASSIFIER_HIDDEN_DIM  - hidden_dim to build the head with, only used as
                           a fallback for a PLAIN state_dict checkpoint
                           (default: 128, matching the original head)
  CLASSIFIER_QUANTIZED   - "true"/"false", only used as a fallback for a
                           plain state_dict (default: false)
  CLASSIFIER_VERSION_TAG - model_version tag prefix (default: joint_v2_postfix)

A structured-pruning or quantized checkpoint from compression/ is saved as
a self-describing {"state_dict", "hidden_dim", "quantized"} dict (see
compression/common.py's load_checkpoint, which this mirrors), so for those
CLASSIFIER_HIDDEN_DIM/CLASSIFIER_QUANTIZED are read automatically from the
file itself and don't need to be set correctly by hand -- avoiding exactly
the kind of silent config-drift bug the DATABASE_URL / load_dotenv() issue
on the v2-agentic-eval branch already taught this project to worry about.
"""
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

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

CLASSIFIER_CHECKPOINT = os.environ.get("CLASSIFIER_CHECKPOINT", "model/classifier_head.pt")
CLASSIFIER_HIDDEN_DIM_FALLBACK = int(os.environ.get("CLASSIFIER_HIDDEN_DIM", str(HIDDEN_DIM)))
CLASSIFIER_QUANTIZED_FALLBACK = os.environ.get("CLASSIFIER_QUANTIZED", "false").lower() == "true"
CLASSIFIER_VERSION_TAG = os.environ.get("CLASSIFIER_VERSION_TAG", "joint_v2_postfix")

ml_state: dict = {}


def _pick_quantized_engine(preferred: str | None = None) -> str:
    supported = torch.backends.quantized.supported_engines
    for candidate in ([preferred] if preferred else []) + ["fbgemm", "qnnpack"]:
        if candidate in supported:
            return candidate
    raise RuntimeError(f"no usable quantized engine found; supported={supported}")


def load_classifier(checkpoint_path: str) -> torch.nn.Module:
    """Builds and loads the classifier head for CLASSIFIER_CHECKPOINT.
    Handles both checkpoint shapes used across this project: a plain
    state_dict (the original trained head, and every unstructured-pruned
    checkpoint from compression/ -- all fixed hidden_dim=128), and the
    self-describing {"state_dict", "hidden_dim", "quantized"} dict that
    structured-pruning and quantization checkpoints save, needed because
    those change the model's actual shape or require converting the module
    to its quantized form BEFORE load_state_dict will accept the packed
    int8 weights.
    """
    loaded = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(loaded, dict) and "state_dict" in loaded and "hidden_dim" in loaded:
        hidden_dim = loaded["hidden_dim"]
        quantized = loaded.get("quantized", False)
        engine = loaded.get("quantization_engine")
        state_dict = loaded["state_dict"]
    else:
        hidden_dim = CLASSIFIER_HIDDEN_DIM_FALLBACK
        quantized = CLASSIFIER_QUANTIZED_FALLBACK
        engine = None
        state_dict = loaded

    classifier = MLPHead(input_dim=1024, hidden_dim=hidden_dim, n_classes=len(LABELS), dropout=DROPOUT)
    if quantized:
        torch.backends.quantized.engine = _pick_quantized_engine(engine)
        classifier = torch.quantization.quantize_dynamic(classifier, {torch.nn.Linear}, dtype=torch.qint8)
    classifier.load_state_dict(state_dict)
    classifier.eval()
    return classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    model, processor, device = load_clip_model()
    ml_state["clip_model"] = model
    ml_state["clip_processor"] = processor
    ml_state["device"] = device

    ml_state["classifier"] = load_classifier(CLASSIFIER_CHECKPOINT)
    ml_state["model_version"] = get_model_version(Path(CLASSIFIER_CHECKPOINT), CLASSIFIER_VERSION_TAG)

    print(f"loaded CLIP on {device}, classifier head from {CLASSIFIER_CHECKPOINT}, "
          f"model_version={ml_state['model_version']}")
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
