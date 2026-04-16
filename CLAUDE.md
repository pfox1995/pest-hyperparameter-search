# CLAUDE.md — Project Context for AI Assistants

## Project Overview

Optuna hyperparameter search for a pest detection Vision-Language model (Qwen3.5-9B LoRA fine-tuning).
Uses a 2-phase strategy: Phase 1 proxy screening (10% data, 50 trials) then Phase 2 full training (10 trials + retrain best).

## Key Architecture

- **hp_search.py** — Main training + HP search script (single file, ~1700 lines)
- **setup_runpod.sh** — RunPod provisioning: env vars, deps, model download, W&B seeding, tmux launch
- **seed_from_wandb.py** — Seeds Optuna study with top trials from previous W&B runs
- **Dataset** — 19-class Korean pest images (AI Hub), hosted on HuggingFace, Unsloth Vision JSONL format

## RunPod Deployment

### File layout on RunPod
```
/workspace/
  hp_search.py          <-- ACTIVE copy (training runs from here)
  hp_search_results.db  <-- Optuna SQLite DB
  hp_search.log         <-- Training log
  data/                 <-- Dataset (downloaded by setup_runpod.sh)
  repo/                 <-- Git clone of this repository
```

### CRITICAL: Deploying code changes
The training runs `/workspace/hp_search.py`, NOT `/workspace/repo/hp_search.py`.
After `git pull` in `/workspace/repo/`, you MUST copy the updated file:
```bash
cd /workspace/repo && git pull
cp /workspace/repo/hp_search.py /workspace/hp_search.py
```
Without this copy, the training will keep running the old version.

### Restarting training
```bash
tmux kill-session -t hp_search 2>/dev/null
tmux new-session -d -s hp_search bash -c "cd /workspace && \
  python3 hp_search.py --proxy --n-trials 50 2>&1 | tee -a /workspace/hp_search.log && \
  python3 hp_search.py --n-trials 10 --retrain 2>&1 | tee -a /workspace/hp_search.log && \
  exec bash"
```

### Monitoring
- `tmux attach -t hp_search` — live view
- `tail -f /workspace/hp_search.log` — log
- Discord webhook receives Korean notifications for each trial
- W&B dashboard: `pfox1995-none/pest-detection-hpsearch`

## Known Issues & Gotchas

### Stale RUNNING trials
When a tmux session is killed mid-trial, Optuna leaves those trials in RUNNING state forever.
hp_search.py now auto-cleans these on startup via raw SQL before `create_study()`.
The cleanup MUST run before Optuna opens the DB, or its CachedStorage layer will revert the changes.

### Smart eval skip (proxy mode)
In proxy mode, non-promising trials (eval_loss > best * 1.05) skip the expensive inference eval.
The fallback `eval_result` dict must include ALL keys that downstream code expects:
`accuracy`, `f1_macro`, `f1_weighted`, `precision_macro`, `recall_macro`, `total`, `per_class`, `confusion_matrix`, `confusion_matrix_path`.

### Optuna SQLite storage
- State column uses VARCHAR(8) with values: 'RUNNING', 'COMPLETE', 'PRUNED', 'FAIL', 'WAITING'
- `study.tell()` does NOT work for trials created by `study.optimize()` — use raw SQL
- `_storage.set_trial_state_values()` API varies across Optuna versions — avoid it

## Code Conventions

- Korean log messages and Discord notifications (target audience is Korean)
- All notifications go through `discord_send()` which is fire-and-forget via background threads
- GPU memory management is aggressive: `try/finally` blocks on every trial with explicit cleanup
- Optuna DB auto-backed up to GitHub on new best trial via `github_upload_db()`
- W&B integration is optional (gracefully disabled if WANDB_API_KEY not set)
