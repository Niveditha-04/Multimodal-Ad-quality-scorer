"""RAG + LLM explanation layer. Given a flagged ad, retrieves grounding
context (rag/retrieve.py) and asks Claude for a short explanation citing it.

Error handling is a most-specific-first chain, not one broad except -- an
invalid key, a rate limit, and a network blip are different failure modes
and a caller (the /score endpoint) may want to react differently, or at
least log differently, than treating them all the same. A failure here
returns a clearly-marked string rather than raising, since an explanation
failure shouldn't take down the classification response that already
succeeded.
"""
import base64
import io
import os

import anthropic
from dotenv import load_dotenv
from PIL import Image

from rag.retrieve import retrieve_context

load_dotenv()

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 300

_client = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set (checked process env and .env)")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


SYSTEM_PROMPT = (
    "You review advertising creatives for a pet-products marketplace. You are "
    "given an ad's image, its text, the category it was flagged under, one or "
    "more policy rules, and similar past examples. Write a short, specific "
    "explanation (2-3 sentences) of why this ad was flagged. Ground the "
    "explanation in the specific policy rule provided -- reference what the "
    "rule actually says, don't just restate the category name. When the flagged "
    "category is 'misleading', the issue is a mismatch between what the image "
    "shows and what the text claims -- look at the image and say what it "
    "actually shows versus what the text describes, don't guess at a reason "
    "that ignores the image. Do not invent claims about the ad that aren't "
    "supported by the image or text given to you."
)


def _image_to_base64_block(image: Image.Image) -> dict:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}


def _build_user_content(ad_text: str, predicted_label: str, context: dict, image: Image.Image) -> list:
    policy_lines = "\n".join(
        f"- {r['category_name']}: {r['rule_description']}" for r in context["policy_rules"]
    )
    example_lines = "\n".join(
        f"- ({e['ground_truth_label']}) {e['ad_text']}" for e in context["examples"]
    ) or "(none retrieved)"

    text_block = {
        "type": "text",
        "text": (
            f'Ad text: "{ad_text}"\n'
            f"Flagged category: {predicted_label}\n\n"
            f"Relevant policy rule(s):\n{policy_lines}\n\n"
            f"Similar past flagged examples (text only, no images available for these):\n{example_lines}\n\n"
            f"The ad's image is attached above. Explain why this ad was flagged."
        ),
    }
    return [_image_to_base64_block(image), text_block]


def generate_explanation(ad_text: str, predicted_label: str, image: Image.Image) -> str:
    context = retrieve_context(ad_text, predicted_label=predicted_label)
    user_content = _build_user_content(ad_text, predicted_label, context, image)

    try:
        response = get_client().messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
    except anthropic.AuthenticationError:
        return "[explanation unavailable: invalid or missing API key]"
    except anthropic.RateLimitError:
        return "[explanation unavailable: rate limited, try again shortly]"
    except anthropic.APIConnectionError:
        return "[explanation unavailable: network error reaching the Anthropic API]"
    except anthropic.APIStatusError as e:
        return f"[explanation unavailable: API error, status {e.status_code}]"
    except RuntimeError as e:
        return f"[explanation unavailable: {e}]"

    if response.stop_reason == "refusal":
        return "[explanation unavailable: model declined to respond]"

    text = next((b.text for b in response.content if b.type == "text"), None)
    if not text:
        return "[explanation unavailable: no text content in model response]"
    return text.strip()
