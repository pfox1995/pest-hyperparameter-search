#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# A6000 Pod Setup — env, deps, data (no HP search, no Optuna)
# ═══════════════════════════════════════════════════════════════════════
# Required RunPod env vars / secrets:
#   HF_TOKEN               — HuggingFace token with dataset access (REQUIRED)
# Optional:
#   WANDB_API_KEY          — enables W&B logging
#   DISCORD_WEBHOOK_URL    — enables Discord progress/failure alerts
#   HF_REPO_ID             — target repo for final model upload
#                            (e.g. "pfox1995/pest-detector-a6000")
#
# Assumes this folder lives inside a cloned copy of the repo:
#   /workspace/repo/final_training_a6000/
#
# Usage:
#   cd /workspace/repo/final_training_a6000
#   bash setup_a6000.sh
#   bash run.sh                              # starts training in tmux
# ═══════════════════════════════════════════════════════════════════════

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ─── Paths ─────────────────────────────────────────────────────────────
export HP_DATA_DIR="${HP_DATA_DIR:-/workspace/data}"
export HP_VOLUME_DIR="${HP_VOLUME_DIR:-/workspace}"
export FINAL_OUTPUT_DIR="${FINAL_OUTPUT_DIR:-${HP_VOLUME_DIR}/best-pest-detector}"

# W&B config (used by train.py)
export WANDB_PROJECT="${WANDB_PROJECT:-pest-detection-hpsearch}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"

# ─── A6000 / Ampere CUDA tuning ────────────────────────────────────────
# expandable_segments reduces fragmentation during long training runs
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ═══════════════════════════════════════════════════════════════════════
echo "=== [0/5] 환경 검증 ==="
# ═══════════════════════════════════════════════════════════════════════

if [[ -z "$HF_TOKEN" ]]; then
    echo "ERROR: HF_TOKEN이 필수입니다 (비공개 데이터셋 접근)."
    echo "RunPod 시크릿에 HF_TOKEN을 설정하세요."
    exit 1
fi

[[ -z "$WANDB_API_KEY" ]]       && echo "NOTE: WANDB_API_KEY 미설정 — W&B 로깅 비활성화"
[[ -z "$DISCORD_WEBHOOK_URL" ]] && echo "NOTE: DISCORD_WEBHOOK_URL 미설정 — Discord 알림 비활성화"
[[ -z "$HF_REPO_ID" ]]          && echo "NOTE: HF_REPO_ID 미설정 — HF Hub 업로드 건너뜀 (LoRA는 로컬 저장)"

# GPU sanity
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA GPU가 감지되지 않습니다'
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
cap = torch.cuda.get_device_capability(0)
print(f'GPU: {name} ({vram:.0f}GB VRAM, sm_{cap[0]}{cap[1]})')
if 'A6000' not in name and vram < 45:
    print(f'WARNING: {name} is not an A6000. HPs were tuned for 48GB Ampere.')
if cap[0] < 8:
    print(f'ERROR: compute capability sm_{cap[0]}{cap[1]} too old (need Ampere sm_80+).')
    import sys; sys.exit(1)
" || { echo "ERROR: GPU 검증 실패"; exit 1; }

# Disk
AVAILABLE_GB=$(( $(df "${HP_VOLUME_DIR}/" 2>/dev/null | awk 'NR==2 {print $4}') / 1024 / 1024 ))
echo "디스크 여유 공간: ${AVAILABLE_GB}GB"
if (( AVAILABLE_GB < 30 )); then
    echo "WARNING: 최소 30GB 권장 (데이터셋 19GB + 체크포인트 + HF 캐시)"
fi

mkdir -p "$FINAL_OUTPUT_DIR"

# ═══════════════════════════════════════════════════════════════════════
echo "=== [1/5] 패키지 설치 ==="
# ═══════════════════════════════════════════════════════════════════════

# Clear stale compiled caches from any previous pod
rm -rf /workspace/unsloth_compiled_cache /root/.cache/unsloth_compiled_cache 2>/dev/null || true

# Korean font for matplotlib (confusion matrix labels)
apt-get update -qq && apt-get install -y -qq fonts-nanum > /dev/null 2>&1 || true
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
    "Pillow>=10.0" \
    "wandb>=0.19" \
    "requests>=2.32" \
    "matplotlib>=3.8" \
    "seaborn>=0.13" \
    "hf_transfer>=0.1" \
    "huggingface_hub>=0.25"

# Flash Attention 2 — 1.5-2× faster on Ampere
pip install flash-attn --no-build-isolation --no-cache-dir 2>&1 | tail -3 \
    || echo "WARNING: Flash Attention 2 설치 실패 (xformers 대체 사용)"

export HF_HUB_ENABLE_HF_TRANSFER=1

# ═══════════════════════════════════════════════════════════════════════
echo "=== [2/5] W&B 로그인 ==="
# ═══════════════════════════════════════════════════════════════════════

if [[ -n "$WANDB_API_KEY" ]]; then
    wandb login "$WANDB_API_KEY" || echo "WARNING: W&B 로그인 실패"
else
    export WANDB_MODE=disabled
fi

# ═══════════════════════════════════════════════════════════════════════
echo "=== [3/5] 데이터셋 다운로드 ==="
# ═══════════════════════════════════════════════════════════════════════

if [ ! -f "${HP_DATA_DIR}/.download_complete" ]; then
    echo "데이터셋 다운로드 시작 (19.4GB)..."
    python3 "$REPO_DIR/download_dataset.py" || {
        echo "ERROR: 데이터셋 다운로드 실패"; exit 1
    }
else
    echo "데이터셋 이미 존재 — 건너뜁니다."
fi

# ═══════════════════════════════════════════════════════════════════════
echo "=== [4/5] 권한 설정 ==="
# ═══════════════════════════════════════════════════════════════════════

chmod +x "$SCRIPT_DIR/run.sh" "$SCRIPT_DIR/watch_golden.sh" 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════════════
echo "=== [5/5] 설정 완료 ==="
# ═══════════════════════════════════════════════════════════════════════

cat <<EOT

✅ A6000 환경 준비 완료.

다음 단계:
    cd $SCRIPT_DIR
    bash run.sh                   # 1 epoch, 전체 val 평가
    EPOCHS=2 bash run.sh          # 에폭 변경
    tmux -u attach -t train       # 학습 진행 확인 (Ctrl+B, D 로 detach)
    tail -f /workspace/_golden/watcher.log   # 최선 체크포인트 추적

출력:
    LoRA adapter:   $FINAL_OUTPUT_DIR/lora/
    Confusion mat:  $FINAL_OUTPUT_DIR/evaluation/
    Golden backup:  /workspace/_golden/best_ckpt/
EOT
