#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# RunPod Setup Script for Pest Detection HP Search
# ═══════════════════════════════════════════════════════════════════════
# 1. Edit the API keys below (rotate them first!)
# 2. Upload both this file and hp_search.py to your RunPod pod
# 3. Run:  bash setup_runpod.sh
# ═══════════════════════════════════════════════════════════════════════

set -eo pipefail  # Exit on error; pipe failures propagate

# ─── API Keys (EDIT THESE — use freshly rotated tokens!) ──────────────
export GITHUB_TOKEN="REPLACE_WITH_GITHUB_PAT"
export WANDB_API_KEY="REPLACE_WITH_WANDB_KEY"
export DISCORD_WEBHOOK_URL="REPLACE_WITH_DISCORD_WEBHOOK"
export HF_TOKEN="REPLACE_WITH_HF_TOKEN"  # Required: dataset is private

# ─── Validate tokens before proceeding ────────────────────────────────
if [[ "$GITHUB_TOKEN" == *"REPLACE"* ]]; then
    echo "ERROR: GITHUB_TOKEN에 실제 토큰을 입력하세요."
    exit 1
fi
if [[ "$WANDB_API_KEY" == *"REPLACE"* ]]; then
    echo "ERROR: WANDB_API_KEY에 실제 키를 입력하세요."
    exit 1
fi
if [[ "$DISCORD_WEBHOOK_URL" == *"REPLACE"* ]]; then
    echo "ERROR: DISCORD_WEBHOOK_URL에 실제 웹훅 URL을 입력하세요."
    exit 1
fi
if [[ "$HF_TOKEN" == *"REPLACE"* ]]; then
    echo "ERROR: HF_TOKEN에 실제 HuggingFace 토큰을 입력하세요. (비공개 데이터셋)"
    exit 1
fi

# ─── RunPod Paths ─────────────────────────────────────────────────────
export HP_DATA_DIR="/workspace/data"
export HP_VOLUME_DIR="/workspace"
export HP_OUTPUT_DIR="${HP_VOLUME_DIR}/hp_search_outputs"
export HP_DB_PATH="${HP_VOLUME_DIR}/hp_search_results.db"
export HP_BEST_MODEL_DIR="${HP_VOLUME_DIR}/best-pest-detector"
export HP_LOG_FILE="${HP_VOLUME_DIR}/hp_search.log"

# ─── W&B Project ──────────────────────────────────────────────────────
export WANDB_PROJECT="pest-detection-hpsearch"
export WANDB_ENTITY=""  # Your W&B username or team (leave empty for default)

# ═══════════════════════════════════════════════════════════════════════
# PRE-FLIGHT CHECKS
# ═══════════════════════════════════════════════════════════════════════

echo "=== [0/5] 사전 점검 ==="

# GPU check
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA GPU가 감지되지 않습니다'
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'GPU: {name} ({vram:.0f}GB)')
" || { echo "ERROR: GPU를 사용할 수 없습니다. Pod 설정을 확인하세요."; exit 1; }

# Disk space check (need ~20GB for dataset + packages)
AVAILABLE_KB=$(df "${HP_DATA_DIR%/*}/" 2>/dev/null | awk 'NR==2 {print $4}' || echo "0")
AVAILABLE_GB=$((AVAILABLE_KB / 1024 / 1024))
if (( AVAILABLE_GB < 15 )); then
    echo "WARNING: ${HP_DATA_DIR}에 ${AVAILABLE_GB}GB만 남아 있습니다. (최소 15GB 권장)"
fi

# Create persistent directories
mkdir -p "$HP_OUTPUT_DIR" "$HP_BEST_MODEL_DIR"

# ═══════════════════════════════════════════════════════════════════════
# INSTALLATION
# ═══════════════════════════════════════════════════════════════════════

echo "=== [1/5] 패키지 설치 중... ==="
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
    "requests>=2.32"

# ─── W&B Login ────────────────────────────────────────────────────────
echo "=== [2/5] W&B 로그인... ==="
if [[ -n "$WANDB_API_KEY" && "$WANDB_API_KEY" != *"REPLACE"* ]]; then
    wandb login "$WANDB_API_KEY" || { echo "ERROR: W&B 로그인 실패"; exit 1; }
else
    echo "W&B 키가 설정되지 않아 건너뜁니다."
    export WANDB_MODE=disabled
fi

# ═══════════════════════════════════════════════════════════════════════
# DATASET DOWNLOAD (idempotent — skips if already present)
# ═══════════════════════════════════════════════════════════════════════

echo "=== [3/5] 데이터셋 확인 중... ==="
if [ ! -f "${HP_DATA_DIR}/.download_complete" ]; then
    echo "데이터셋 다운로드 시작..."
    python3 -c "
from huggingface_hub import snapshot_download
import os
token = os.environ.get('HF_TOKEN') or None
snapshot_download(
    'Himedia-AI-01/pest-detection-korean',
    repo_type='dataset',
    local_dir='${HP_DATA_DIR}',
    local_dir_use_symlinks=False,
    token=token,
)
print('데이터셋 다운로드 완료!')
" || { echo "ERROR: 데이터셋 다운로드 실패"; exit 1; }
    touch "${HP_DATA_DIR}/.download_complete"
else
    echo "데이터셋이 이미 존재합니다. 다운로드를 건너뜁니다."
fi

# Verify key files exist
if [ ! -f "${HP_DATA_DIR}/train.jsonl" ]; then
    echo "ERROR: train.jsonl이 없습니다. 데이터셋을 확인하세요."
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════
# START TMUX SESSION (survives SSH disconnect)
# ═══════════════════════════════════════════════════════════════════════

echo "=== [4/5] 학습 세션 준비 중... ==="

# Install tmux if not available
if ! command -v tmux &> /dev/null; then
    apt-get update -qq && apt-get install -y -qq tmux > /dev/null 2>&1 || true
fi

SESSION_NAME="hp_search"

# Kill existing session if present
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

echo "=== [5/5] 하이퍼파라미터 검색 시작! ==="
echo ""
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
echo ""

# Start training in tmux session
tmux new-session -d -s "$SESSION_NAME" bash -c "
    cd /workspace

    echo '=== Phase 1: Proxy 검색 시작 ==='
    python3 hp_search.py --proxy --n-trials 50 2>&1 | tee -a '${HP_LOG_FILE}'
    PHASE1_EXIT=\$?

    if [ \$PHASE1_EXIT -ne 0 ]; then
        echo 'Phase 1 실패 (exit code: '\$PHASE1_EXIT'). Phase 2를 건너뜁니다.'
        exit \$PHASE1_EXIT
    fi

    echo '=== Phase 2: Full 검색 + 재학습 시작 ==='
    python3 hp_search.py --n-trials 10 --retrain 2>&1 | tee -a '${HP_LOG_FILE}'

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
