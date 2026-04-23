#!/usr/bin/env python3
"""FastAPI serving for pest-detector-final (merged bf16).

Speed+accuracy knobs applied:
  - stop_strings=["\n", "<|im_end|>"]  → kills trailing chat tokens at source
  - do_sample=False (greedy)           → deterministic, cacheable
  - torch.inference_mode()             → lighter than no_grad
  - image resize to MAX_IMAGE_DIM=768  → matches training distribution
  - functools.lru_cache on _predict    → dedup repeated image hashes
  - PREFIX normalization fallback      → rescue off-vocab generations
  - warmup at lifespan startup         → no cold-start on first real request

Env:
  MODEL_DIR        = /workspace/pest-detector-merged   (from merge_adapter.py)
  MAX_IMAGE_DIM    = 768                                (resize cap)
  LRU_CACHE_SIZE   = 256                                (same-image dedup)
  SERVE_LOG_LEVEL  = INFO
"""
import base64
import hashlib
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig


# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════
MODEL_DIR      = Path(os.environ.get("MODEL_DIR", "/workspace/pest-detector-merged"))
MAX_IMAGE_DIM  = int(os.environ.get("MAX_IMAGE_DIM", "768"))
LRU_CACHE_SIZE = int(os.environ.get("LRU_CACHE_SIZE", "256"))
MAX_BATCH      = int(os.environ.get("MAX_BATCH", "16"))
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE          = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Must match the system prompt used at TRAINING time. Any drift here
# degrades accuracy — do not edit without retraining.
SYSTEM_MSG = (
    "당신은 작물 해충 식별 전문가입니다. "
    "사진을 보고 해충의 이름만 한국어로 답하세요. "
    '해충이 없으면 "정상"이라고만 답하세요. '
    "부가 설명 없이 이름만 출력하세요."
)

# User-turn template. Training used 5 variations at random; picking the
# first (and most common) one here keeps inference in-distribution.
USER_PROMPT = "이 사진에 있는 해충의 이름을 알려주세요."

logging.basicConfig(
    level=os.environ.get("SERVE_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("serve")


# ═══════════════════════════════════════════════════════════════════
# Globals set in lifespan
# ═══════════════════════════════════════════════════════════════════
model = None
processor = None
gen_cfg: GenerationConfig | None = None
classes: list[str] = []
classes_sorted_desc: list[str] = []   # pre-sorted longest-first for PREFIX match


# ═══════════════════════════════════════════════════════════════════
# Lifespan: load + warmup
# ═══════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, processor, gen_cfg, classes, classes_sorted_desc

    if not MODEL_DIR.exists():
        raise RuntimeError(
            f"MODEL_DIR does not exist: {MODEL_DIR}. "
            f"Run `python merge_adapter.py` first."
        )

    t0 = time.time()
    log.info(f"Loading merged model from {MODEL_DIR} ({DTYPE}, {DEVICE})...")
    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_DIR), dtype=DTYPE, device_map=DEVICE,
    ).eval()
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    log.info(f"Model loaded in {time.time() - t0:.1f}s")

    # classes.json (optional — enables PREFIX normalization)
    cls_path = MODEL_DIR / "classes.json"
    if cls_path.exists():
        classes = json.loads(cls_path.read_text(encoding="utf-8"))
        classes_sorted_desc = sorted(classes, key=len, reverse=True)
        log.info(f"Loaded {len(classes)} classes for PREFIX normalization")
    else:
        log.warning(
            f"{cls_path} missing — running without PREFIX normalization. "
            f"Raw model output will be returned."
        )

    gen_cfg = GenerationConfig(
        max_new_tokens=16,
        do_sample=False,
        use_cache=True,
        # stop_strings needs the tokenizer handed to .generate() too.
        stop_strings=["\n", "<|im_end|>"],
        # pad_token fallback — some Qwen tokenizers don't set it
        pad_token_id=(
            processor.tokenizer.pad_token_id
            or processor.tokenizer.eos_token_id
        ),
    )

    # Warmup — triggers all compile/alloc paths before taking traffic.
    log.info("Warmup forward...")
    t0 = time.time()
    _warmup_img = Image.new("RGB", (512, 512), color=(128, 160, 96))
    _ = _predict_raw(_img_to_hash_key(_warmup_img), _warmup_img)
    log.info(f"Warmup done in {time.time() - t0:.1f}s. Ready for traffic.")

    yield

    log.info("Shutdown.")


app = FastAPI(
    title="Pest Detector Serving",
    description="Qwen3.5-9B + pfox1995/pest-detector-final LoRA (merged bf16)",
    version="1.0.0",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════
# API schemas
# ═══════════════════════════════════════════════════════════════════
class PredictRequest(BaseModel):
    image_base64: str = Field(..., description="base64-encoded JPEG or PNG")


class PredictResponse(BaseModel):
    pest: str = Field(..., description="Normalized class name (or raw if off-vocab)")
    raw: str = Field(..., description="Exact model output")
    known_class: bool = Field(..., description="True iff pest ∈ classes.json")
    latency_ms: int


class BatchPredictRequest(BaseModel):
    images_base64: list[str] = Field(..., max_length=MAX_BATCH)


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    classes_count: int
    model_dir: str


# ═══════════════════════════════════════════════════════════════════
# Prediction core
# ═══════════════════════════════════════════════════════════════════
def _resize_image(img: Image.Image) -> Image.Image:
    """Cap longest edge at MAX_IMAGE_DIM. Matches training preprocessing."""
    w, h = img.size
    longest = max(w, h)
    if longest <= MAX_IMAGE_DIM:
        return img
    scale = MAX_IMAGE_DIM / longest
    return img.resize(
        (int(round(w * scale)), int(round(h * scale))),
        resample=Image.Resampling.LANCZOS,
    )


def _img_to_hash_key(img: Image.Image) -> str:
    """Stable hash of image pixels — used as LRU cache key."""
    # tobytes() after resize gives deterministic key regardless of source format
    img = _resize_image(img.convert("RGB"))
    h = hashlib.sha256()
    h.update(img.tobytes())
    h.update(f"{img.size}".encode())
    return h.hexdigest()


def _normalize(raw: str) -> tuple[str, bool]:
    """PREFIX match raw against known classes; fall back to raw."""
    g = raw.strip()
    for c in classes_sorted_desc:
        if g.startswith(c):
            return c, True
    return g, False


@lru_cache(maxsize=LRU_CACHE_SIZE)
def _predict_raw(_hash_key: str, img: Image.Image | None = None) -> tuple[str, str, bool]:
    """Run one inference. LRU-cached on image hash.

    Returns (pest, raw, known_class).
    `img` is only used on cache miss; on hit, the tuple is returned directly.
    """
    assert img is not None, "img must be provided on cache miss"
    img = _resize_image(img.convert("RGB"))

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text": USER_PROMPT},
        ]},
    ]
    tmpl = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        img, tmpl,
        add_special_tokens=False, return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            generation_config=gen_cfg,
            tokenizer=processor.tokenizer,  # required for stop_strings
        )

    raw = processor.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    pest, known = _normalize(raw)
    return pest, raw, known


def _predict_one(img: Image.Image) -> tuple[str, str, bool]:
    """Public single-image path with LRU caching by image hash."""
    key = _img_to_hash_key(img)
    return _predict_raw(key, img)


# ═══════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════
def _decode_image(b64: str) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64, validate=True)))
        return img.convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    img = _decode_image(req.image_base64)
    t0 = time.time()
    pest, raw, known = _predict_one(img)
    return PredictResponse(
        pest=pest, raw=raw, known_class=known,
        latency_ms=int((time.time() - t0) * 1000),
    )


@app.post("/predict_batch", response_model=BatchPredictResponse)
async def predict_batch(req: BatchPredictRequest):
    """Sequential fan-out (per-request GPU batching would require a
    custom worker loop; keeping serial-per-request for simplicity and
    because LRU cache hits make repeat-image batches fast anyway)."""
    results: list[PredictResponse] = []
    for b64 in req.images_base64:
        img = _decode_image(b64)
        t0 = time.time()
        pest, raw, known = _predict_one(img)
        results.append(PredictResponse(
            pest=pest, raw=raw, known_class=known,
            latency_ms=int((time.time() - t0) * 1000),
        ))
    return BatchPredictResponse(results=results)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if model is not None else "loading",
        model_loaded=model is not None,
        device=DEVICE,
        classes_count=len(classes),
        model_dir=str(MODEL_DIR),
    )


@app.get("/cache_stats")
async def cache_stats():
    """Peek at the LRU hit rate — useful for sizing LRU_CACHE_SIZE."""
    info = _predict_raw.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "maxsize": info.maxsize,
        "currsize": info.currsize,
        "hit_rate": (info.hits / max(1, info.hits + info.misses)),
    }
