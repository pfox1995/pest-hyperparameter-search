#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# RunPod Setup — Phase 2 RESUME (fetch DB, retrain best trial, upload to HF)
# ═══════════════════════════════════════════════════════════════════════
# Use when:
#   - You already have a completed (or partial) Phase 1 SQLite DB backed up
#     at https://raw.githubusercontent.com/<GITHUB_REPO>/main/hp_search_results.db
#   - You want to skip Phase 1 AND the Phase 2 trial loop entirely, and just
#     retrain the current best trial (study.best_trial) on 100% data, then
#     publish the adapter to HuggingFace.
#   - If you want to run N additional full-fidelity trials first (to confirm
#     the proxy winner), change `--n-trials 0` to `--n-trials N` in step [8/8].
#
# Required RunPod env vars / secrets:
#   WANDB_API_KEY          — W&B API key
#   HF_TOKEN               — HuggingFace token (must have write access to HF_REPO_ID)
#   DISCORD_WEBHOOK_URL    — Discord webhook (optional)
#   github_pat             — GitHub PAT (optional; only needed if DB backup during Phase 2 is wanted)
#
# Optional RunPod env vars:
#   HF_REPO_ID             — HuggingFace repo to publish adapter (default: pfox1995/qwen35-9b-pest-detector-lora)
#   HF_PRIVATE             — "true"/"false" (default: true)
#   GITHUB_REPO            — source repo for DB download (default: pfox1995/pest-hyperparameter-search)
#   WANDB_PROJECT          — W&B project name
#   HP_DATA_DIR / HP_VOLUME_DIR — path overrides
#
# Usage:
#   bash setup_runpod_phase2.sh
# ═══════════════════════════════════════════════════════════════════════

set -eo pipefail

# ─── Map RunPod env vars ──────────────────────────────────────────────
export GITHUB_TOKEN="${GITHUB_TOKEN:-${github_pat:-}}"

export HP_DATA_DIR="${HP_DATA_DIR:-/workspace/data}"
export HP_VOLUME_DIR="${HP_VOLUME_DIR:-/workspace}"
export HP_OUTPUT_DIR="${HP_OUTPUT_DIR:-${HP_VOLUME_DIR}/hp_search_outputs}"
export HP_DB_PATH="${HP_DB_PATH:-${HP_VOLUME_DIR}/hp_search_results.db}"
export HP_BEST_MODEL_DIR="${HP_BEST_MODEL_DIR:-${HP_VOLUME_DIR}/best-pest-detector}"
export HP_LOG_FILE="${HP_LOG_FILE:-${HP_VOLUME_DIR}/hp_search.log}"

export WANDB_PROJECT="${WANDB_PROJECT:-pest-detection-hpsearch}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"

export GITHUB_REPO="${GITHUB_REPO:-pfox1995/pest-hyperparameter-search}"
export HF_REPO_ID="${HF_REPO_ID:-pfox1995/qwen35-9b-pest-detector-lora}"
export HF_PRIVATE="${HF_PRIVATE:-true}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ═══════════════════════════════════════════════════════════════════════
# [0/8] 환경 검증
# ═══════════════════════════════════════════════════════════════════════

echo "=== [0/8] 환경 검증 ==="

MISSING=""
[[ -z "$WANDB_API_KEY" ]]       && MISSING="${MISSING}  WANDB_API_KEY (선택)\n"
[[ -z "$HF_TOKEN" ]]            && MISSING="${MISSING}  HF_TOKEN (필수: 데이터셋 접근 + 모델 업로드)\n"
[[ -z "$DISCORD_WEBHOOK_URL" ]] && MISSING="${MISSING}  DISCORD_WEBHOOK_URL (선택)\n"

if [[ -n "$MISSING" ]]; then
    echo "WARNING: 다음 환경 변수가 설정되지 않았습니다:"
    echo -e "$MISSING"
fi

if [[ -z "$HF_TOKEN" ]]; then
    echo "ERROR: HF_TOKEN이 필수입니다 (비공개 데이터셋 + 모델 업로드). RunPod 시크릿을 확인하세요."
    exit 1
fi

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

AVAILABLE_KB=$(df "${HP_VOLUME_DIR}/" 2>/dev/null | awk 'NR==2 {print $4}' || echo "0")
AVAILABLE_GB=$((AVAILABLE_KB / 1024 / 1024))
echo "디스크 여유 공간: ${AVAILABLE_GB}GB"
if (( AVAILABLE_GB < 25 )); then
    echo "WARNING: Phase 2(전체 데이터) + 모델 업로드에는 최소 25GB 권장"
fi

mkdir -p "$HP_OUTPUT_DIR" "$HP_BEST_MODEL_DIR"

echo ""
echo "  HP_VOLUME_DIR     = $HP_VOLUME_DIR"
echo "  HP_DB_PATH        = $HP_DB_PATH"
echo "  GITHUB_REPO (DB)  = $GITHUB_REPO"
echo "  HF_REPO_ID        = $HF_REPO_ID  (private=$HF_PRIVATE)"
echo "  WANDB_PROJECT     = $WANDB_PROJECT"
echo "  HF_TOKEN          = $([ -n "$HF_TOKEN" ] && echo '***set***' || echo 'MISSING')"
echo "  GITHUB_TOKEN      = $([ -n "$GITHUB_TOKEN" ] && echo '***set***' || echo 'MISSING')"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# [1/8] 패키지 설치
# ═══════════════════════════════════════════════════════════════════════

echo "=== [1/8] 패키지 설치 중... ==="

rm -rf /workspace/unsloth_compiled_cache 2>/dev/null || true
rm -rf /root/.cache/unsloth_compiled_cache 2>/dev/null || true

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
    "optuna>=4.0,<5.0" \
    "plotly>=5.0" \
    "kaleido>=0.2" \
    "wandb>=0.19" \
    "requests>=2.32" \
    "matplotlib>=3.8" \
    "seaborn>=0.13" \
    "hf_transfer>=0.1" \
    "huggingface_hub>=0.26"

pip install flash-attn --no-build-isolation --no-cache-dir 2>&1 | tail -3 || echo "WARNING: Flash Attention 2 설치 실패"

export HF_HUB_ENABLE_HF_TRANSFER=1

# ─── W&B + HF logins ──────────────────────────────────────────────────
echo "=== [2/8] W&B + HuggingFace 로그인... ==="
if [[ -n "$WANDB_API_KEY" ]]; then
    wandb login "$WANDB_API_KEY" || echo "WARNING: W&B 로그인 실패 (계속 진행)"
else
    export WANDB_MODE=disabled
fi

python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])" \
    || { echo "ERROR: HuggingFace 로그인 실패"; exit 1; }

# ═══════════════════════════════════════════════════════════════════════
# [3/8] 데이터셋 + 모델 다운로드
# ═══════════════════════════════════════════════════════════════════════

echo "=== [3/8] 데이터셋 확인 중... ==="
if [ ! -f "${HP_DATA_DIR}/.download_complete" ]; then
    echo "데이터셋 다운로드 시작 (19.4GB)..."
    python3 download_dataset.py || { echo "ERROR: 데이터셋 다운로드 실패"; exit 1; }
fi

if [ ! -f "${HP_DATA_DIR}/train.jsonl" ]; then
    echo "ERROR: train.jsonl이 없습니다."
    exit 1
fi

echo "=== [3b/8] 모델 사전 다운로드 (Qwen3.5-9B)... ==="
python3 -c "
from huggingface_hub import snapshot_download
import os
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
path = snapshot_download('unsloth/Qwen3.5-9B', ignore_patterns=['*.gguf'])
print(f'Model cached at: {path}')
" 2>&1 | tail -5 || echo "WARNING: 모델 사전 다운로드 실패"

# ═══════════════════════════════════════════════════════════════════════
# [4/8] GitHub에서 Phase 1 DB 다운로드
# ═══════════════════════════════════════════════════════════════════════

echo "=== [4/8] Phase 1 Optuna DB 다운로드 중... ==="
DB_URL="https://raw.githubusercontent.com/${GITHUB_REPO}/main/hp_search_results.db"
echo "  원본: ${DB_URL}"
echo "  대상: ${HP_DB_PATH}"

# If a DB already exists (re-run), back it up before overwriting
if [ -f "$HP_DB_PATH" ]; then
    BK="${HP_DB_PATH}.pre_phase2.$(date +%s).bak"
    cp "$HP_DB_PATH" "$BK"
    echo "  기존 DB 백업: $BK"
fi

HTTP_CODE=$(curl -sSL -w '%{http_code}' -o "$HP_DB_PATH" "$DB_URL")
if [ "$HTTP_CODE" != "200" ]; then
    echo "ERROR: DB 다운로드 실패 (HTTP ${HTTP_CODE}). GitHub 백업을 확인하세요."
    exit 1
fi

DB_BYTES=$(stat -c '%s' "$HP_DB_PATH")
echo "  다운로드 완료: ${DB_BYTES} bytes"

# Sanity check: verify it's a valid SQLite DB and has completed trials
python3 <<PYEOF
import optuna, sys
try:
    s = optuna.load_study(study_name='pest-detection-hpsearch', storage='sqlite:///${HP_DB_PATH}')
    states = {}
    for t in s.trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1
    completed = [t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    print(f"  Study 상태: {states}")
    print(f"  COMPLETE trials: {len(completed)}")
    if completed:
        best = min(completed, key=lambda t: t.value)
        print(f"  최적: #{best.number}, eval_loss={best.value:.6f}")
    if len(completed) < 3:
        print("  WARNING: COMPLETE 트라이얼이 3개 미만입니다. Phase 2가 의미있을지 재고하세요.")
        sys.exit(0)
except Exception as e:
    print(f"ERROR: DB 검증 실패: {e}")
    sys.exit(1)
PYEOF

# ═══════════════════════════════════════════════════════════════════════
# [5/8] 학습 세션 준비
# ═══════════════════════════════════════════════════════════════════════

echo "=== [5/8] 학습 세션 준비 중... ==="

if ! command -v tmux &> /dev/null; then
    apt-get update -qq && apt-get install -y -qq tmux > /dev/null 2>&1 || true
fi

SESSION_NAME="hp_search"
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

echo "=== [6/8] 환경 변수 ==="
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  HF_REPO_ID=$HF_REPO_ID  (private=$HF_PRIVATE)"

echo ""
echo "=== [7/8] Phase 2 시작! ==="
echo ""
echo "  Phase 2: 최적 트라이얼 재학습 (전체 데이터, 1회)"
echo "           - Proxy top-3 트라이얼이 동일한 아키텍처로 수렴 → 단일 재학습으로 단축"
echo "  Phase 3: GitHub Release + HuggingFace 어댑터 업로드"
echo ""
echo "  tmux 세션에서 실행됩니다. SSH 연결이 끊어져도 계속됩니다."
echo ""
echo "  진행 확인:"
echo "    tmux attach -t ${SESSION_NAME}"
echo "    tail -f ${HP_LOG_FILE}"
echo "    Ctrl+B, D                 # 세션에서 나가기"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# [8/8] tmux에서 Phase 2 + 업로드 실행
# ═══════════════════════════════════════════════════════════════════════

tmux new-session -d -s "$SESSION_NAME" bash -c "
    cd /workspace

    echo '=== Phase 2: 최적 트라이얼 (#6) 재학습 시작 ==='
    # --n-trials 0 = skip optimization loop, go straight to retrain_best(study)
    # Proxy top-3 shared same architecture; trial #6 had lowest eval_loss. Skipping
    # 10 extra full-fidelity trials saves ~40 GPU-hours with negligible accuracy risk.
    python3 hp_search.py --n-trials 0 --retrain 2>&1 | tee -a '${HP_LOG_FILE}'
    PHASE2_EXIT=\$?

    if [ \$PHASE2_EXIT -ne 0 ]; then
        echo 'Phase 2 실패 (exit code: '\$PHASE2_EXIT'). 업로드를 건너뜁니다.'
        exec bash
    fi

    LORA_PATH='${HP_BEST_MODEL_DIR}/lora'
    EVAL_DIR='${HP_BEST_MODEL_DIR}/evaluation'
    METRICS_JSON=\"\$EVAL_DIR/metrics.json\"

    if [ ! -d \"\$LORA_PATH\" ]; then
        echo 'WARNING: 저장된 LoRA 어댑터가 없습니다. 업로드 건너뜀: '\"\$LORA_PATH\"
        exec bash
    fi

    echo ''
    echo '=== Phase 3a: GitHub Release 업로드 ==='
    python3 -c \"
from hp_search import github_upload_results
import json, os

eval_result = None
if os.path.exists('\$METRICS_JSON'):
    with open('\$METRICS_JSON') as f:
        eval_result = json.load(f)

url = github_upload_results(eval_result=eval_result)
if url:
    print(f'GitHub Release: {url}')
else:
    print('GitHub release upload skipped or failed')
\" 2>&1 | tee -a '${HP_LOG_FILE}'

    echo ''
    echo '=== Phase 3b: HuggingFace 업로드 ==='
    HF_REPO_ID='$HF_REPO_ID' HF_PRIVATE='$HF_PRIVATE' \\
    HP_BEST_MODEL_DIR='$HP_BEST_MODEL_DIR' \\
    python3 hf_upload.py 2>&1 | tee -a '${HP_LOG_FILE}' || \\
        echo 'WARNING: HuggingFace 업로드 실패 (로컬 모델은 ${HP_BEST_MODEL_DIR}/lora에 보존됨)'

    echo ''
    echo '=== 모든 작업 완료! ==='
    echo '최적 모델 (로컬): ${HP_BEST_MODEL_DIR}/lora'
    echo 'HuggingFace:     https://huggingface.co/${HF_REPO_ID}'
    echo ''
    exec bash
"

echo "tmux 세션 '${SESSION_NAME}'에서 Phase 2가 시작되었습니다."
echo "연결: tmux attach -t ${SESSION_NAME}"
