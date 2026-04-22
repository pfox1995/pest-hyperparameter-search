#!/usr/bin/env bash
# A6000 final-retrain launcher.
# Mirrors setup_runpod.sh's exit-code safety pattern.
#
# CRITICAL: 'set -o pipefail' + ${PIPESTATUS[0]} are required so that a
# python crash is detected. Without these, `python | tee` returns tee's
# exit code (always 0 if log is writable) — run zhgoxqpx reported
# "=== EXIT: 0 ===" after crashing.
#
# Usage (from the pod, inside the cloned repo):
#   cd /workspace/repo/final_training_a6000
#   bash run.sh                                   # 1 epoch, full val eval
#   EPOCHS=2 bash run.sh                          # 2 epochs
#   EXTRA_ARGS='--no-wandb --no-eval' bash run.sh
set -eo pipefail

SESSION="${SESSION_NAME:-train}"
LOG="${LOG_FILE:-/workspace/final_train.log}"
EPOCHS="${EPOCHS:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="$SCRIPT_DIR/train.py"
WATCH_SH="$SCRIPT_DIR/watch_golden.sh"

if [ ! -f "$TRAIN_PY" ]; then
    echo "ERROR: train.py not found at $TRAIN_PY"; exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists. Kill it first:"
    echo "  tmux kill-session -t $SESSION"
    exit 1
fi

# ─── Launch training ───────────────────────────────────────────────────
tmux new-session -d -s "$SESSION" bash -c "
    set -o pipefail
    cd /workspace

    notify_failure() {
        local exit_code=\"\$1\"
        local tail_lines
        tail_lines=\$(tail -n 30 '$LOG' 2>/dev/null | sed 's/\"/\\\\\"/g' | tr '\n' '\\\\n' | head -c 1500)
        if [ -n \"\${DISCORD_WEBHOOK_URL:-}\" ]; then
            curl -sS -H 'Content-Type: application/json' -X POST \\
                -d '{\"content\":\"💥 **A6000 최종 학습 비정상 종료** (exit='\"\$exit_code\"', SIGKILL/OOM 가능성)\\n\`\`\`'\"\$tail_lines\"'\`\`\`\"}' \\
                \"\$DISCORD_WEBHOOK_URL\" >/dev/null 2>&1 || true
        fi
        echo \"학습 실패 (exit code: \$exit_code)\"
    }

    echo '=== A6000 final retrain 시작 (epochs=$EPOCHS) ==='
    HP_LAZY_DATASET=1 python3 -u '$TRAIN_PY' --epochs $EPOCHS $EXTRA_ARGS 2>&1 | tee '$LOG'
    EXIT=\${PIPESTATUS[0]}

    if [ \$EXIT -ne 0 ]; then
        notify_failure \$EXIT
        exec bash
    fi

    echo '=== A6000 final retrain 완료 ==='
    exec bash
"

# ─── Launch golden-checkpoint sidecar ──────────────────────────────────
if [ -f "$WATCH_SH" ] && ! tmux has-session -t golden 2>/dev/null; then
    tmux new-session -d -s golden "bash '$WATCH_SH'"
    echo "Golden-checkpoint sidecar started in tmux 'golden'."
fi

echo ""
echo "Training started in tmux session '$SESSION'."
echo "  Attach:  tmux -u attach -t $SESSION"
echo "  Detach:  Ctrl+B, D"
echo "  Log:     $LOG"
echo "  Golden:  tail -f /workspace/_golden/watcher.log"
