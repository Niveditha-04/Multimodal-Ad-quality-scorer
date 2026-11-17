"""Optional MCP wrapper around the /score functionality, so an MCP-capable
agent can call ad compliance checking as a tool directly, without going
through the HTTP API.

Separate from api/main.py and does not import it. Both entry points use the
same shared modules (model/clip_features.py, model/classifier_architecture.py,
rag/explain.py), so they score identically without depending on each other
being up.
"""
import io

import torch
from mcp.server.mcpserver import MCPServer
from PIL import Image, UnidentifiedImageError

from db.models import Ad, Score
from db.session import SessionLocal, init_db
from model.clip_features import load_clip_model, embed_batch, build_joint_features
from model.classifier_architecture import LABELS, MLPHead, HIDDEN_DIM, DROPOUT
from model.model_version import get_model_version

server = MCPServer(
    name="ad-quality-scorer",
    instructions=(
        "Scores a pet-product ad creative (image + text) for policy violations "
        "and quality issues using a multimodal classifier. Portfolio project, "
        "not a production moderation system."
    ),
)

_state: dict = {}


def _ensure_loaded() -> None:
    if _state:
        return
    init_db()
    model, processor, device = load_clip_model()
    classifier = MLPHead(input_dim=1024, hidden_dim=HIDDEN_DIM, n_classes=len(LABELS), dropout=DROPOUT)
    classifier.load_state_dict(torch.load("model/classifier_head.pt", map_location="cpu"))
    classifier.eval()
    _state["clip_model"] = model
    _state["clip_processor"] = processor
    _state["device"] = device
    _state["classifier"] = classifier
    _state["model_version"] = get_model_version()


@server.tool()
def check_ad_compliance(image_path: str, ad_text: str) -> dict:
    """Scores one ad creative for policy violations and quality issues.

    Args:
        image_path: path to a local image file (jpg/png) for the ad creative.
        ad_text: the ad's copy text.

    Returns:
        predicted_label (one of approved/policy_violation/misleading/low_quality),
        confidence (0-1), model_version, and explanation (present only when
        the ad was flagged -- an LLM-generated, policy-grounded explanation,
        or null if ANTHROPIC_API_KEY isn't configured).
    """
    _ensure_loaded()

    # Expected input problems (missing file, empty text) return a structured
    # {"error": ...} dict instead of raising. A raised ValueError is wrapped by
    # the SDK into an UnexpectedToolError with a full traceback, which is harder
    # for a calling agent to read than an error message in the tool output.
    if not ad_text or not ad_text.strip():
        return {"error": "ad_text must not be empty"}

    try:
        with open(image_path, "rb") as f:
            pil_image = Image.open(io.BytesIO(f.read())).convert("RGB")
    except FileNotFoundError:
        return {"error": f"image_path not found: {image_path}"}
    except UnidentifiedImageError:
        return {"error": f"file at image_path is not a readable image: {image_path}"}

    image_embeds, text_embeds = embed_batch(
        _state["clip_model"], _state["clip_processor"], _state["device"],
        [pil_image], [ad_text],
    )
    features = build_joint_features(image_embeds, text_embeds)

    with torch.no_grad():
        logits = _state["classifier"](torch.tensor(features, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])

    predicted_label = LABELS[pred_idx]

    session = SessionLocal()
    try:
        ad = Ad(image_path=f"<mcp:{image_path}>", ad_text=ad_text, ground_truth_label=None)
        session.add(ad)
        session.flush()
        session.add(Score(
            ad_id=ad.id, predicted_label=predicted_label,
            confidence=confidence, model_version=_state["model_version"],
        ))
        session.commit()
        ad_id = ad.id
    finally:
        session.close()

    explanation = None
    if predicted_label != "approved":
        from rag.explain import generate_explanation
        explanation = generate_explanation(ad_text=ad_text, predicted_label=predicted_label, image=pil_image)

    return {
        "ad_id": ad_id,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "model_version": _state["model_version"],
        "explanation": explanation,
    }


if __name__ == "__main__":
    server.run(transport="stdio")
