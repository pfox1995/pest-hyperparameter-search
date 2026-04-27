#!/usr/bin/env bash
# Serve the merged Q5_K_M pest-detection model via llama-server.
# Exposes an OpenAI-compatible /v1/chat/completions endpoint on :8080.

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
# Qwen3.5 defaults to "thinking mode"; for 19-class pest labels we want
# the final answer immediately. The system prompt below disables thinking.
exec "${LLAMA_DIR}/build/bin/llama-server" \
    --model         "${MODEL}" \
    --mmproj        "${MMPROJ}" \
    --ctx-size      8192 \
    --n-gpu-layers  999 \
    --temp          0.0 \
    --top-k         1 \
    --port          "${PORT}" \
    --host          0.0.0.0 \
    --jinja
