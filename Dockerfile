# FastAPI service for POST /score and GET /stats. CLASSIFIER_CHECKPOINT selects
# the classifier head at run time (see the docstring in api/main.py), so one
# image serves the baseline head or the compressed checkpoint.
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch wheels, since the image has no GPU. Installed first so pip does
# not replace them with the CUDA build when it resolves requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 \
      --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Download the CLIP weights and Chroma's default embedding model at build time,
# so containers start without network access.
RUN python3 -c "\
from transformers import CLIPModel, CLIPProcessor; \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')" \
    && python3 -c "\
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2; \
ONNXMiniLM_L6_V2()(['warm up'])"

# Code and model files used at request time.
COPY api/ api/
COPY db/models.py db/session.py db/
COPY model/classifier_architecture.py model/clip_features.py model/model_version.py model/classifier_head.pt model/
COPY rag/__init__.py rag/retrieve.py rag/explain.py rag/
COPY compression/checkpoints/ compression/checkpoints/

# Build the retrieval index (policy rules plus training-split examples) from the
# dataset, then delete the temporary database so the container starts with an
# empty scores table.
COPY data/dataset.csv data/split.json data/
COPY db/load_dataset.py db/policy_categories.py db/
COPY rag/build_vector_store.py rag/
RUN python -m db.load_dataset && python -m rag.build_vector_store && rm -f db/ads.db

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
