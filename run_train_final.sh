#!/usr/bin/env bash
# Launcher for train_final.py — mirrors setup_runpod.sh's exit-code safety.
#
# CRITICAL: 'set -o pipefail' + ${PIPESTATUS[0]} are required so that a
# python crash (incl. SIGKILL from OOM killer) is detected. Without these,
# 'python | tee' returns tee's exit code (always 0 if log is writable) and
# a crash silently reports as success — which is exactly what happened in
# W&B run zhgoxqpx (the "=== EXIT: 0 ===" bug).
#
# Usage (on the pod):
#   cd /workspace
#   bash run_train_final.sh           # defaults: 1 epoch, session=train
#   EPOCHS=2 bash run_train_final.sh  # override
#
# After launch: tmux -u attach -t train   (Ctrl+B, D to detach)
set -eo pipefail

SESSION="${SESSION_NAME:-train}"
LOG="${LOG_FILE:-/workspace/final_train.log}"
EPOCHS="${EPOCHS:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists. Kill it first:"
    echo "  tmux kill-session -t $SESSION"
    exit 1
fi

tmux new-session -d -s "$SESSION" bash -c "
    set -o pipefail
    cd /workspace

    notify_failure() {
        local exit_code=\"\$1\"
        local tail_lines
        tail_lines=\$(tail -n 30 '$LOG' 2>/dev/null | sed 's/\"/\\\\\"/g' | tr '\n' '\\\\n' | head -c 1500)
        if [ -n \"\${DISCORD_WEBHOOK_URL:-}\" ]; then
            curl -sS -H 'Content-Type: application/json' -X POST \\
                -d '{\"content\":\"💥 **최종 재학습 비정상 종료** (exit='\"\$exit_code\"', SIGKILL/OOM 가능성)\\n\`\`\`'\"\$tail_lines\"'\`\`\`\"}' \\
                \"\$DISCORD_WEBHOOK_URL\" >/dev/null 2>&1 || true
        fi
        echo \"학습 실패 (exit code: \$exit_code)\"
    }

    echo '=== Final retrain 시작 (epochs=$EPOCHS) ==='
    HP_LAZY_DATASET=1 python3 -u train_final.py --epochs $EPOCHS $EXTRA_ARGS 2>&1 | tee '$LOG'
    EXIT=\${PIPESTATUS[0]}

    if [ \$EXIT -ne 0 ]; then
        notify_failure \$EXIT
        exec bash
    fi

    echo '=== Final retrain 완료 ==='
    exec bash
"

echo "Training started in tmux session '$SESSION'."
echo "  Attach:  tmux -u attach -t $SESSION"
echo "  Detach:  Ctrl+B, D"
echo "  Log:     $LOG"
