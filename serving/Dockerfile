# Production serving image for pest-detector-final.
# Base: RunPod's PyTorch 2.8 / CUDA 12.8 image (matches what we validated in eval).
# Build on the pod AFTER running merge_adapter.py (model dir is COPY'd in).

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Korean font for consistent rendering if you ever extend serve.py to
# generate per-request visualizations. Safe to remove if not needed.
RUN mkdir -p /usr/share/fonts/truetype/nanum \
 && curl -sSL -o /usr/share/fonts/truetype/nanum/NanumGothic.ttf \
      https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf \
 && fc-cache -fv

WORKDIR /app

# Python deps first — changes less often than code, keeps layers cached.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip uninstall -y peft    # peft only needed at merge time, not at serve time

COPY serve.py .

# Merged model — MUST be built by merge_adapter.py before `docker build`.
# ~18 GB, so the container image is huge. For RunPod, consider mounting
# the merged model as a volume instead and deleting this COPY line.
COPY pest-detector-merged /app/model

ENV MODEL_DIR=/app/model \
    MAX_IMAGE_DIM=768 \
    LRU_CACHE_SIZE=256 \
    SERVE_LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1

EXPOSE 8000

# Single worker: the model lives on one GPU, multiple uvicorn workers would
# each try to load it → OOM. Scale horizontally by running multiple pods
# behind a load balancer, not multiple workers per pod.
CMD ["uvicorn", "serve:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
