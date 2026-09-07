# Serves the FastAPI /score and /stats endpoints. Which classifier
# checkpoint gets loaded is a runtime choice (CLASSIFIER_CHECKPOINT env var,
# see api/main.py's own docstring for the full list) -- this one image
# serves either the original trained head or any pruned/quantized
# checkpoint from the compression/ study, so there's no separate
# "optimized" image to build or drift out of sync with the regular one.
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch/torchvision wheels. This project's classifier head and
# CLIP backbone run on CPU here regardless of what device they use on a
# dev machine (Docker has no MPS passthrough, and this image assumes no
# GPU host) -- the CUDA-enabled default wheels would add several hundred
# MB for code paths this image never exercises. Installed before the rest
# of requirements.txt so pip's resolver sees torch/torchvision already
# satisfied at import time and doesn't pull the CUDA build back in as a
# transitive dependency.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 \
      --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download both models this image needs at REQUEST time into the
# image itself, at BUILD time instead. Found by actually running this
# container and hitting /score for real: both CLIPModel.from_pretrained
# (~600MB, transformers/HF Hub) and chromadb's default embedding function
# (~79MB ONNX MiniLM, used by rag/retrieve.py's collection.query calls)
# download themselves lazily on first use if not already cached -- fine on
# a dev machine that already has them cached from earlier runs, but inside
# a fresh container this means every cold start needs reliable outbound
# network access, and in this environment the chromadb download timed out
# mid-transfer and took the whole /score request down with it (visible in
# `docker logs`, not simulated). Baking both caches into the image at
# build time removes that runtime dependency entirely and makes cold
# starts fast and deterministic instead of network-dependent.
RUN python3 -c "\
from transformers import CLIPModel, CLIPProcessor; \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')" \
    && python3 -c "\
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2; \
ONNXMiniLM_L6_V2()(['warm up'])"

# Only what api/main.py's import chain and the RAG retrieval path actually
# touch at request time -- not model/train_classifier.py, model/results/,
# db/queries.sql, rag/build_vector_store.py, ab_test/ (none of these are
# imported by api/main.py), and not data/images/ (the 44MB training photo
# set /score never reads; a request's image comes from the upload itself).
# compression/checkpoints/ ships every pruned/quantized checkpoint from the
# study so CLASSIFIER_CHECKPOINT can select any of them at `docker run`
# time without rebuilding the image.
COPY api/ api/
COPY db/models.py db/session.py db/
COPY model/classifier_architecture.py model/clip_features.py model/model_version.py model/classifier_head.pt model/
COPY rag/__init__.py rag/retrieve.py rag/explain.py rag/
COPY rag/chroma_db/ rag/chroma_db/
COPY compression/checkpoints/ compression/checkpoints/
COPY .env.example .

# db/ads.db is intentionally NOT copied in -- SQLAlchemy's init_db() creates
# the schema fresh on first startup (see db/session.py), so the container
# starts with an empty scores table rather than baking in this dev
# machine's local data.
RUN mkdir -p db

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
