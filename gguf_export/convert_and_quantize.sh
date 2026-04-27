#!/usr/bin/env bash
# Build llama.cpp + convert merged FP16 -> GGUF + quantize to Q5_K_M.
# Run on RunPod after merge_lora.py has produced /workspace/merged_fp16.
#
# Disk required during run:
#   /workspace/merged_fp16          ~17.9 GB (input, can delete after step 3)
#   /workspace/gguf/model-F16.gguf  ~17.9 GB (intermediate, delete after step 4)
#   /workspace/gguf/pest-Q5_K_M.gguf ~6.6 GB (final, KEEP)
#   /workspace/gguf/mmproj-F16.gguf  ~1.4 GB (final, KEEP — vision tower)
# Peak: ~37 GB. Final: ~8 GB.

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
LLAMA_DIR="${WORKDIR}/llama.cpp"
MERGED_DIR="${WORKDIR}/merged_fp16"
GGUF_DIR="${WORKDIR}/gguf"
F16_GGUF="${GGUF_DIR}/Qwen3.5-9B-pest-F16.gguf"
Q5_GGUF="${GGUF_DIR}/Qwen3.5-9B-pest-Q5_K_M.gguf"
MMPROJ="${GGUF_DIR}/mmproj-F16.gguf"

mkdir -p "${GGUF_DIR}"

# ── 1. Build llama.cpp (latest master — needed for Gated DeltaNet ops) ──
if [ ! -d "${LLAMA_DIR}" ]; then
    echo "[1/5] Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp "${LLAMA_DIR}"
else
    echo "[1/5] Updating llama.cpp..."
    cd "${LLAMA_DIR}" && git pull && cd -
fi

echo "[2/5] Building llama.cpp with CUDA..."
apt-get update -qq && apt-get install -y -qq \
    pciutils build-essential cmake curl libcurl4-openssl-dev
cmake "${LLAMA_DIR}" -B "${LLAMA_DIR}/build" \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
cmake --build "${LLAMA_DIR}/build" --config Release -j "$(nproc)"

# Python deps for the conversion script
pip install -q -r "${LLAMA_DIR}/requirements.txt"

# ── 3. Convert merged HF -> FP16 GGUF ──
echo "[3/5] Converting merged HF model to FP16 GGUF..."
python "${LLAMA_DIR}/convert_hf_to_gguf.py" "${MERGED_DIR}" \
    --outfile "${F16_GGUF}" \
    --outtype f16

# ── 4. Quantize FP16 GGUF -> Q5_K_M ──
echo "[4/5] Quantizing to Q5_K_M..."
"${LLAMA_DIR}/build/bin/llama-quantize" \
    "${F16_GGUF}" \
    "${Q5_GGUF}" \
    Q5_K_M

# ── 5. Fetch the mmproj vision tower from Unsloth's pre-built repo ──
# (Vision encoder doesn't change with LoRA on language layers — we can
# reuse Unsloth's official F16 mmproj instead of regenerating it.)
echo "[5/5] Downloading mmproj vision tower..."
huggingface-cli download unsloth/Qwen3.5-9B-GGUF mmproj-F16.gguf \
    --local-dir "${GGUF_DIR}"

echo ""
echo "[done] Final files:"
ls -lh "${Q5_GGUF}" "${MMPROJ}"
echo ""
echo "Reclaim disk by removing intermediates when satisfied:"
echo "  rm -rf ${MERGED_DIR}    # ~17.9 GB"
echo "  rm    ${F16_GGUF}       # ~17.9 GB"
