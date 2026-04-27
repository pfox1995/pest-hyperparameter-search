#!/usr/bin/env bash
# Serve the merged Q5_K_M pest-detection model via llama-server.
# Exposes an OpenAI-compatible /v1/chat/completions endpoint on :8080.
#
# CRITICAL — thinking-mode flags
#   The training in hp_search.py:620 used `apply_chat_template(messages,
#   add_generation_prompt=True)` with NO `enable_thinking` argument. On
#   Qwen3.5-9B (small tier) the default is non-thinking, so the LoRA was
#   trained to emit just `<label><|im_end|>` with no `<think>` block.
#
#   But on current llama.cpp builds (b8227+), Qwen3.5-9B GGUF defaults
#   thinking ON via the chat template, breaking inference for any LoRA
#   trained on the non-thinking distribution. Symptoms: model gets stuck
#   in <think>...</think>, content stays empty, accuracy collapses.
#   Refs: ggml-org/llama.cpp#20833, #20409, #20516, #20789.
#
#   Fix: pass --reasoning off + --reasoning-format none + --reasoning-budget 0
#   AND have clients send chat_template_kwargs={"enable_thinking":false} per
#   request as defense-in-depth.
#
# Verify in startup log: `srv init: chat template, thinking = 0`.

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
LLAMA_DIR="${WORKDIR}/llama.cpp"
GGUF_DIR="${WORKDIR}/gguf"
MODEL="${GGUF_DIR}/Qwen3.5-9B-pest-Q5_K_M.gguf"
MMPROJ="${GGUF_DIR}/mmproj-F16.gguf"
PORT="${PORT:-8080}"

[ -f "${MODEL}"  ] || { echo "Missing: ${MODEL}";  exit 1; }
[ -f "${MMPROJ}" ] || { echo "Missing: ${MMPROJ}"; exit 1; }

# --temp 0.0 + --top-k 1: deterministic for classification.
exec "${LLAMA_DIR}/build/bin/llama-server" \
    --model              "${MODEL}" \
    --mmproj             "${MMPROJ}" \
    --jinja \
    --reasoning          off \
    --reasoning-format   none \
    --reasoning-budget   0 \
    --ctx-size           8192 \
    --n-gpu-layers       999 \
    --temp               0.0 \
    --top-k              1 \
    --port               "${PORT}" \
    --host               0.0.0.0 \
    --alias              qwen3.5-9b-pest
