#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# RunPod Setup Script for Pest Detection HP Search
# ═══════════════════════════════════════════════════════════════════════
# All API keys are read from RunPod environment variables (set in pod config).
# Required RunPod env vars / secrets:
#   WANDB_API_KEY          — W&B API key (secret)
#   DISCORD_WEBHOOK_URL    — Discord webhook (secret)
#   HF_TOKEN               — HuggingFace token (secret)
#   github_pat             — GitHub PAT (secret)
# Optional RunPod env vars:
#   WANDB_PROJECT          — W&B project name
#   HP_DATA_DIR            — Dataset path
#   HP_VOLUME_DIR          — Persistent volume
#
# Usage:
#   1. Set secrets + env vars in RunPod pod config
#   2. Upload this file + hp_search.py + seed_from_wandb.py to /workspace
#   3. Run:  bash setup_runpod.sh
# ═══════════════════════════════════════════════════════════════════════

set -eo pipefail

# ─── Map RunPod env vars to what the code expects ─────────────────────
# RunPod provides these directly; we only set defaults for derived paths.
# github_pat (RunPod secret name) → GITHUB_TOKEN (what code uses)
export GITHUB_TOKEN="${GITHUB_TOKEN:-${github_pat:-}}"

# Paths — use RunPod env vars if set, otherwise defaults
export HP_DATA_DIR="${HP_DATA_DIR:-/workspace/data}"
export HP_VOLUME_DIR="${HP_VOLUME_DIR:-/workspace}"
export HP_OUTPUT_DIR="${HP_OUTPUT_DIR:-${HP_VOLUME_DIR}/hp_search_outputs}"
export HP_DB_PATH="${HP_DB_PATH:-${HP_VOLUME_DIR}/hp_search_results.db}"
export HP_BEST_MODEL_DIR="${HP_BEST_MODEL_DIR:-${HP_VOLUME_DIR}/best-pest-detector}"
export HP_LOG_FILE="${HP_LOG_FILE:-${HP_VOLUME_DIR}/hp_search.log}"

# W&B — use RunPod env var if set
export WANDB_PROJECT="${WANDB_PROJECT:-pest-detection-hpsearch}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"

# GitHub repo for DB backup (owner/repo format)
export GITHUB_REPO="${GITHUB_REPO:-pfox1995/pest-hyperparameter-search}"

# HuggingFace repo for final model upload (optional, not used by default)
export HF_REPO_ID="${HF_REPO_ID:-}"

# ─── CUDA / PyTorch tuning for A6000 (48GB) ──────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ═══════════════════════════════════════════════════════════════════════
# VALIDATE ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════

echo "=== [0/7] 환경 검증 ==="

# Check required secrets
MISSING=""
[[ -z "$WANDB_API_KEY" ]]        && MISSING="${MISSING}  WANDB_API_KEY\n"
[[ -z "$HF_TOKEN" ]]             && MISSING="${MISSING}  HF_TOKEN\n"
[[ -z "$DISCORD_WEBHOOK_URL" ]]  && MISSING="${MISSING}  DISCORD_WEBHOOK_URL (선택)\n"
[[ -z "$GITHUB_TOKEN" ]]         && MISSING="${MISSING}  GITHUB_TOKEN / github_pat (선택)\n"

if [[ -n "$MISSING" ]]; then
    echo "WARNING: 다음 환경 변수가 설정되지 않았습니다:"
    echo -e "$MISSING"
    echo "(필수 변수가 없으면 해당 기능이 비활성화됩니다)"
fi

# Hard fail only on truly required vars
if [[ -z "$HF_TOKEN" ]]; then
    echo "ERROR: HF_TOKEN이 필수입니다 (비공개 데이터셋 접근). RunPod 시크릿을 확인하세요."
    exit 1
fi

# GPU check
python3 -c "
import torch, os
assert torch.cuda.is_available(), 'CUDA GPU가 감지되지 않습니다'
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024**3
print(f'GPU: {name} ({vram:.0f}GB VRAM)')
print(f'RAM: {ram:.0f}GB')
print(f'CUDA: {torch.version.cuda}')
print(f'PyTorch: {torch.__version__}')
" || { echo "ERROR: GPU를 사용할 수 없습니다."; exit 1; }

# Disk space check
AVAILABLE_KB=$(df "${HP_VOLUME_DIR}/" 2>/dev/null | awk 'NR==2 {print $4}' || echo "0")
AVAILABLE_GB=$((AVAILABLE_KB / 1024 / 1024))
echo "디스크 여유 공간: ${AVAILABLE_GB}GB"
if (( AVAILABLE_GB < 15 )); then
    echo "WARNING: 최소 15GB 권장"
fi

# Create persistent directories
mkdir -p "$HP_OUTPUT_DIR" "$HP_BEST_MODEL_DIR"

# Print resolved config
echo ""
echo "  HP_DATA_DIR       = $HP_DATA_DIR"
echo "  HP_VOLUME_DIR     = $HP_VOLUME_DIR"
echo "  HP_DB_PATH        = $HP_DB_PATH"
echo "  WANDB_PROJECT     = $WANDB_PROJECT"
echo "  GITHUB_REPO       = $GITHUB_REPO"
echo "  PYTORCH_CUDA_ALLOC_CONF = $PYTORCH_CUDA_ALLOC_CONF"
echo "  WANDB_API_KEY     = $([ -n "$WANDB_API_KEY" ] && echo '***set***' || echo 'MISSING')"
echo "  GITHUB_TOKEN      = $([ -n "$GITHUB_TOKEN" ] && echo '***set***' || echo 'MISSING')"
echo "  DISCORD_WEBHOOK   = $([ -n "$DISCORD_WEBHOOK_URL" ] && echo '***set***' || echo 'MISSING')"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# INSTALLATION
# ═══════════════════════════════════════════════════════════════════════

echo "=== [1/7] 패키지 설치 중... ==="

# Clear stale unsloth compiled cache from previous pods/CUDA versions
rm -rf /workspace/unsloth_compiled_cache 2>/dev/null || true
rm -rf /root/.cache/unsloth_compiled_cache 2>/dev/null || true

# Install Korean font for matplotlib confusion matrix labels
apt-get update -qq && apt-get install -y -qq fonts-nanum > /dev/null 2>&1 || true
# Clear matplotlib font cache so it discovers the new font
rm -rf /root/.cache/matplotlib 2>/dev/null || true

pip install --upgrade pip
pip install \
    "unsloth[cu128-torch280]" \
    "transformers>=5.2,<6.0" \
    "trl==0.22.2" \
    "datasets>=3.0,<4.0" \
    "Pillow>=10.0" \
    "accelerate>=1.0,<2.0" \
    "scikit-learn>=1.4" \
    "optuna>=4.0,<5.0" \
    "plotly>=5.0" \
    "kaleido>=0.2" \
    "wandb>=0.19" \
    "requests>=2.32" \
    "matplotlib>=3.8" \
    "seaborn>=0.13" \
    "hf_transfer>=0.1"

# Flash Attention 2 — 1.5-2x faster training on Ampere+ GPUs (A6000, A100, H100)
# --no-cache-dir avoids cross-device link errors on RunPod (pip cache vs /tmp on different filesystems)
pip install flash-attn --no-build-isolation --no-cache-dir 2>&1 | tail -3 || echo "WARNING: Flash Attention 2 설치 실패 (xformers 대체 사용)"

# Enable Rust-based parallel download accelerator for HuggingFace (3-5x faster)
export HF_HUB_ENABLE_HF_TRANSFER=1

# ─── W&B Login ────────────────────────────────────────────────────────
echo "=== [2/7] W&B 로그인... ==="
if [[ -n "$WANDB_API_KEY" ]]; then
    wandb login "$WANDB_API_KEY" || { echo "ERROR: W&B 로그인 실패"; exit 1; }
else
    echo "W&B 키가 설정되지 않아 건너뜁니다."
    export WANDB_MODE=disabled
fi

# ═══════════════════════════════════════════════════════════════════════
# DATASET DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════

echo "=== [3/7] 데이터셋 확인 중... ==="
if [ ! -f "${HP_DATA_DIR}/.download_complete" ]; then
    echo "데이터셋 다운로드 시작 (19.4GB)..."
    python3 download_dataset.py || { echo "ERROR: 데이터셋 다운로드 실패"; exit 1; }
else
    echo "데이터셋이 이미 존재합니다. 다운로드를 건너뜁니다."
fi

if [ ! -f "${HP_DATA_DIR}/train.jsonl" ]; then
    echo "ERROR: train.jsonl이 없습니다. 데이터셋을 확인하세요."
    exit 1
fi

# Pre-download the base model so Trial 0 doesn't waste time on it
echo "=== [3b/7] 모델 사전 다운로드 (Qwen3.5-9B)... ==="
python3 -c "
from huggingface_hub import snapshot_download
import os
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
path = snapshot_download('unsloth/Qwen3.5-9B', ignore_patterns=['*.gguf'])
print(f'Model cached at: {path}')
" 2>&1 | tail -5 || echo "WARNING: 모델 사전 다운로드 실패 (첫 트라이얼에서 다운로드됨)"

# ═══════════════════════════════════════════════════════════════════════
# SEED FROM W&B
# ═══════════════════════════════════════════════════════════════════════

echo "=== [4/7] W&B에서 이전 결과 시딩 중... ==="
if [[ -n "$WANDB_API_KEY" ]]; then
    python3 seed_from_wandb.py --top-n 5 --add-neighbors 2>&1 | tee -a "${HP_LOG_FILE}" || {
        echo "WARNING: W&B 시딩 실패. 새로운 검색으로 계속합니다."
    }
else
    echo "W&B 키가 없어 시딩을 건너뜁니다."
fi

# ═══════════════════════════════════════════════════════════════════════
# START TMUX SESSION
# ═══════════════════════════════════════════════════════════════════════

echo "=== [5/7] 학습 세션 준비 중... ==="

if ! command -v tmux &> /dev/null; then
    apt-get update -qq && apt-get install -y -qq tmux > /dev/null 2>&1 || true
fi

SESSION_NAME="hp_search"
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

echo "=== [6/7] 환경 변수 확인 ==="
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  WANDB_PROJECT=$WANDB_PROJECT"
echo "  GITHUB_REPO=$GITHUB_REPO"

echo "=== [7/7] 하이퍼파라미터 검색 시작! ==="
echo ""
echo "  Phase 0: W&B 시딩 완료 (이전 최적 파라미터 + 변형 enqueued)"
echo "  Phase 1: Proxy 검색 (50 trials, 10% 데이터, 1 에폭)"
echo "  Phase 2: Full 검색 (10 trials, 전체 데이터) + 최적 모델 재학습"
echo ""
echo "  tmux 세션에서 실행됩니다. SSH 연결이 끊어져도 계속됩니다."
echo ""
echo "  진행 확인:"
echo "    tmux attach -t ${SESSION_NAME}     # 세션 연결"
echo "    tail -f ${HP_LOG_FILE}             # 로그 확인"
echo "    Ctrl+B, D                          # 세션에서 나가기"
echo ""
echo "  Discord 알림이 자동으로 전송됩니다."
echo "  Optuna DB가 GitHub에 자동 백업됩니다."
echo ""

# Start training in tmux session
tmux new-session -d -s "$SESSION_NAME" bash -c "
    cd /workspace

    echo '=== Phase 1: Proxy 검색 시작 (seeded trials run first) ==='
    python3 hp_search.py --proxy --n-trials 50 2>&1 | tee -a '${HP_LOG_FILE}'
    PHASE1_EXIT=\$?

    if [ \$PHASE1_EXIT -ne 0 ]; then
        echo 'Phase 1 실패 (exit code: '\$PHASE1_EXIT'). Phase 2를 건너뜁니다.'
        exit \$PHASE1_EXIT
    fi

    echo '=== Phase 2: Full 검색 + 재학습 시작 ==='
    python3 hp_search.py --n-trials 10 --retrain 2>&1 | tee -a '${HP_LOG_FILE}'

    echo '=== Phase 3: GitHub 릴리스 업로드 ==='
    python3 -c \"
from hp_search import github_upload_results
import json, os

eval_result = None
metrics_path = os.path.join(
    os.environ.get('HP_BEST_MODEL_DIR', '/workspace/best-pest-detector'),
    'evaluation', 'metrics.json'
)
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        eval_result = json.load(f)

url = github_upload_results(eval_result=eval_result)
if url:
    print(f'GitHub Release: {url}')
else:
    print('GitHub release upload skipped or failed')
\" 2>&1 | tee -a '${HP_LOG_FILE}'

    echo ''
    echo '=== 모든 작업 완료! ==='
    echo '최적 모델: ${HP_BEST_MODEL_DIR}/lora'
    echo '결과: ${HP_OUTPUT_DIR}/'
    echo ''
    echo '이 터미널은 열려 있습니다. 결과를 확인한 후 exit으로 닫으세요.'
    exec bash
"

echo "tmux 세션 '${SESSION_NAME}'에서 학습이 시작되었습니다."
echo "연결: tmux attach -t ${SESSION_NAME}"
