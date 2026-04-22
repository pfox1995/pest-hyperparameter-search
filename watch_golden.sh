#!/usr/bin/env bash
# Golden-checkpoint backup sidecar.
#
# Runs alongside train_final.py. Whenever HF Trainer updates
# best_model_checkpoint in trainer_state.json, copy that checkpoint to
# /workspace/_golden/best_ckpt so we have a safe copy outside the
# save_total_limit rotation window. Belt-and-suspenders against:
#   - checkpoint rotation bugs (esp. with stale checkpoint-* dirs)
#   - accidental output_dir overwrites
#   - pod crash mid-run (copy is already persisted)
#
# Launch in its own tmux session:
#   tmux new-session -d -s golden 'bash /workspace/watch_golden.sh'
#
# Inspect progress:
#   tail -f /workspace/_golden/watcher.log
set -u
WATCH_DIR="${WATCH_DIR:-/workspace/best-pest-detector}"
GOLDEN="${GOLDEN_DIR:-/workspace/_golden}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

mkdir -p "$GOLDEN"
LOG="$GOLDEN/watcher.log"
SRC_MARKER="$GOLDEN/best_source.txt"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher started; watching $WATCH_DIR" | tee -a "$LOG"

while true; do
    # Most-recent checkpoint holds the freshest trainer_state.json
    latest_ckpt=$(ls -td "$WATCH_DIR"/checkpoint-*/ 2>/dev/null | head -1)
    if [ -z "$latest_ckpt" ] || [ ! -f "$latest_ckpt/trainer_state.json" ]; then
        sleep "$POLL_INTERVAL"
        continue
    fi

    best_path=$(python3 -c "
import json, sys
try:
    d = json.load(open('$latest_ckpt/trainer_state.json'))
    print(d.get('best_model_checkpoint') or '')
except Exception:
    sys.exit(0)
" 2>/dev/null)

    if [ -z "$best_path" ] || [ ! -d "$best_path" ]; then
        sleep "$POLL_INTERVAL"
        continue
    fi

    last_copied=$(cat "$SRC_MARKER" 2>/dev/null || echo "")
    if [ "$best_path" != "$last_copied" ]; then
        ts=$(date -u +%H:%M:%S)
        loss=$(python3 -c "
import json
d = json.load(open('$latest_ckpt/trainer_state.json'))
print(f\"{d.get('best_metric', float('nan')):.6f}\")
" 2>/dev/null || echo "?")
        echo "[$ts] new best: $(basename "$best_path") (eval_loss=$loss)" | tee -a "$LOG"
        rm -rf "$GOLDEN/best_ckpt.tmp"
        cp -r "$best_path" "$GOLDEN/best_ckpt.tmp"
        rm -rf "$GOLDEN/best_ckpt"
        mv "$GOLDEN/best_ckpt.tmp" "$GOLDEN/best_ckpt"
        echo "$best_path" > "$SRC_MARKER"
    fi

    sleep "$POLL_INTERVAL"
done
