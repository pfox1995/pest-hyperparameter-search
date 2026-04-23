# Pest Detector Serving

Production inference server for [`pfox1995/pest-detector-final`](https://huggingface.co/pfox1995/pest-detector-final) LoRA adapter on `unsloth/Qwen3.5-9B` base. FastAPI + bf16 `transformers` — no Unsloth required at serve time.

Optimized for **best accuracy × speed** trade-off on a single RunPod A40 (48 GB VRAM, $0.22/hr spot):

- **`stop_strings=["\n"]`** in generation — kills the `\nassistant\n<think>` trailing noise at the source (fixes the 36% vs 95% strict-vs-prefix gap observed in eval).
- **`merge_and_unload()`** at deploy time — folds LoRA deltas into base weights, removes PEFT from the serving critical path (~20% faster forward pass, simpler dependency tree).
- **Greedy decoding (`do_sample=False`)** — deterministic outputs, safe to cache by image hash.
- **Image pre-resize to 768 px** — matches training-time `MAX_IMAGE_DIM` so inputs stay in-distribution; cuts ViT forward time on oversized inputs.
- **`torch.inference_mode()`** — slightly lighter than `torch.no_grad()` since it skips version counter bumps.
- **Warmup at startup** — first call compiles CUDA kernels / caches; we trigger a dummy forward during FastAPI `lifespan` so the first real request isn't a 30-second outlier.
- **Optional LRU cache on `_predict`** — same image → cached response, zero GPU time (enabled by default, size 256).

## Topology

```
merge_adapter.py  →  /workspace/pest-detector-merged/   (~18 GB bf16, no PEFT)
                     /workspace/pest-detector-merged/classes.json

serve.py          →  reads merged dir, serves HTTP :8000
```

## Deploy

### 1. Pick a pod

**RTX A40, 48 GB, $0.22/hr spot.** That's 2× the memory you need for bf16 Qwen3.5-9B + KV cache + batch headroom. Avoid A5000 (24 GB — no room for concurrent requests) and A100 (80 GB — paying for memory you won't use).

### 2. Provision

```bash
cd serving
pip install -r requirements.txt
python merge_adapter.py      # ~2 min, reuses HF cache if you eval'd earlier
```

This produces `/workspace/pest-detector-merged/` containing the merged model + processor + a `classes.json` extracted from the adapter's training metrics (if `/workspace/hf_eval_v*/metrics.json` exists; otherwise uses a hardcoded fallback list).

### 3. Start the server

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000 --workers 1
```

First ~60 s: weight load (~18 GB mmap) + warmup forward. After that: ~0.7–1.2 req/sec on A40, bf16 eager mode, single stream.

### 4. Call it

```bash
curl -X POST http://pod:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_base64":"'$(base64 -w0 pest.jpg)'"}'
```

```json
{
  "pest": "배추흰나비",
  "raw": "배추흰나비",
  "known_class": true,
  "latency_ms": 980
}
```

See [`client_example.py`](client_example.py) for a Python client that batches a folder of images.

## API

### `POST /predict`

**Request**
```json
{ "image_base64": "<base64-encoded JPEG or PNG>" }
```

**Response**
```json
{
  "pest": "<normalized class name (or raw if off-vocab)>",
  "raw": "<exact model output>",
  "known_class": true,
  "latency_ms": 980
}
```

- `pest`: longest-matching class from `classes.json`, falls back to stripped raw output.
- `known_class`: `true` iff `pest ∈ classes.json`. If `false`, the model emitted something off-vocabulary — treat it as "uncertain" in your downstream logic.

### `GET /health`

Returns `{"status": "ok", "model_loaded": true}` once warmup finishes. Use as your RunPod health-check endpoint.

### `POST /predict_batch`

Takes a list of up to 16 base64 images, returns a list of predictions. Uses GPU batching internally — ~4× throughput vs sequential single-image calls at batch size 8 on A40.

## Performance notes

| Knob | Default | Why |
|---|---|---|
| `max_new_tokens` | 16 | Pest names are 2-7 chars; 16 covers all plus room for stop token |
| `do_sample` | False | Greedy → deterministic → cacheable |
| `use_cache` | True | KV cache; free speedup |
| `num_beams` | 1 | No beam search — accuracy gain tiny, cost 3× |
| `stop_strings` | `["\n", "<\|im_end\|>"]` | Stops generation at the real EOS; prevents chat-template trailing tokens |
| LRU cache size | 256 | Same image hash → cached response |
| Request timeout | 30 s | A40 at eager bf16 rarely exceeds 2 s; anything longer is broken |

## What's intentionally NOT done

- **No 4-bit quantization.** A40 has 48 GB — you don't need it. Bf16 preserves the exact accuracy measured in eval.
- **No torch.compile.** Unsloth's `for_inference` compile path broke 50% of eval samples on Blackwell; plain eager is boring but bulletproof.
- **No vLLM/TGI.** Qwen3-Next's hybrid linear_attn isn't in any serving runtime yet (as of 2026-01). Plain transformers + FastAPI is your only option today.
- **No beam search.** Accuracy ceiling on 20-class short-output classification is already hit by greedy; beams just cost 3× GPU time.

## What you SHOULD do before launching

1. Verify `classes.json` matches your actual training labels (merge script auto-populates from `metrics.json` if present; otherwise it's a fallback list that may be out of sync with your adapter).
2. Set a real `HF_TOKEN` env var if your base model repo ever goes private.
3. Put nginx/caddy in front with a proper rate limit + TLS (uvicorn's built-in is fine for dev, not prod).
4. Monitor `latency_ms` in your logs — sustained >1.5 s on A40 means something's wrong (OOM, thermal throttle, new image shape triggering recompile).
