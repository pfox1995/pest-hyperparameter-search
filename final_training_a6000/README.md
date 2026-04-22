# Final Training on RTX A6000

Self-contained deploy folder for running the final retrain of the pest detection VLM on an RTX A6000 (48 GB, Ampere sm_86). Frozen Optuna trial 015 hyperparameters with stability hardening applied after the run zhgoxqpx divergence.

## What's in here

| File | Purpose |
|---|---|
| `train.py` | Training script — loads model, runs trainer, evaluates, pushes to HF Hub |
| `setup_a6000.sh` | One-shot pod provisioning — env + deps + dataset download |
| `run.sh` | Safe launcher — pipefail + Discord alerts, starts training + sidecar in tmux |
| `watch_golden.sh` | Sidecar that snapshots the best checkpoint to `/workspace/_golden/` |

Imports shared helpers (dataset loading, `evaluate_model`, etc.) from the parent `hp_search.py`. Keep this folder inside a cloned copy of the repo.

## Quick start

### 1. Provision the pod

Launch a RunPod instance with:
- **GPU:** RTX A6000 (48 GB VRAM)
- **Volume:** ≥ 50 GB (19 GB dataset + ~20 GB base model cache + checkpoints)
- **Secrets set in pod config:**
  - `HF_TOKEN` (required — dataset is gated)
  - `HF_REPO_ID` (recommended — final model upload, e.g. `pfox1995/pest-detector-a6000`)
  - `WANDB_API_KEY` (optional — enables W&B run tracking)
  - `DISCORD_WEBHOOK_URL` (optional — failure/completion alerts)

### 2. Clone repo and run setup

```bash
cd /workspace
git clone https://github.com/pfox1995/pest-hyperparameter-search.git repo
cd repo/final_training_a6000
bash setup_a6000.sh
```

`setup_a6000.sh` installs Unsloth + Transformers + TRL, downloads the 19 GB dataset, logs into W&B. Takes ~15 minutes.

### 3. Launch training

```bash
bash run.sh                    # 1 epoch (default), full val set eval
EPOCHS=2 bash run.sh           # 2 epochs
EXTRA_ARGS='--no-wandb' bash run.sh
```

Training runs in tmux session `train`. Golden-checkpoint sidecar runs in tmux session `golden`. Both survive disconnects.

### 4. Monitor

```bash
tmux -u attach -t train                        # live view (Ctrl+B, D to detach)
tail -f /workspace/final_train.log             # log
tail -f /workspace/_golden/watcher.log         # best-checkpoint tracker
```

### 5. Outputs

After training completes (~3 hours for 1 epoch):

| Artifact | Location |
|---|---|
| Final LoRA adapter | `/workspace/best-pest-detector/lora/` |
| Confusion matrix PNG | `/workspace/best-pest-detector/evaluation/confusion_matrix_trial_a6000-final.png` |
| Per-class metrics JSON | `/workspace/best-pest-detector/evaluation/metrics.json` |
| Model card | `/workspace/best-pest-detector/README.md` |
| Golden checkpoint backup | `/workspace/_golden/best_ckpt/` |
| HF Hub | `https://huggingface.co/<HF_REPO_ID>` (private) |

## Differences vs parent `train_final.py`

1. **Stability fixes baked in** (not flags):
   - `optim="adamw_torch"` (not `adamw_8bit` — the 8-bit optimizer caused divergence in run zhgoxqpx)
   - `warmup_ratio=0.03` (softens step-0 LR shock)
   - `save_steps=25` + `eval_steps=25` (captures pre-divergence window, `load_best_model_at_end` pins best)
2. **A6000 memory adjustments:**
   - `per_device_eval_batch_size=2` (halved from 4 for 48 GB headroom)
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (reduces fragmentation)
3. **Teacher feedback applied:**
   - `--eval-samples` defaults to full val set (~1595), not 500. Produces a reliable 19-class confusion matrix (~84 samples/class).
4. **Cleaner output directory scheme** — run name includes `a6000` tag in W&B.

## Troubleshooting

**OOM during training:** Unlikely with these settings, but if it happens, lower `HP["max_seq_length"]` from 1024 to 768 in `train.py`. Don't change batch size — it would require re-tuning the LR.

**Flash Attention install fails:** The `setup_a6000.sh` degrades gracefully to xformers. Training works either way; flash-attn is just ~1.5× faster.

**HF upload fails at end:** Local LoRA is safe at `/workspace/best-pest-detector/lora/`. Re-run manually:
```bash
python3 -c "
from unsloth import FastVisionModel
from huggingface_hub import upload_folder
import os
model, tokenizer = FastVisionModel.from_pretrained('/workspace/best-pest-detector/lora')
model.push_to_hub(os.environ['HF_REPO_ID'], token=os.environ['HF_TOKEN'], private=True)
upload_folder(folder_path='/workspace/best-pest-detector/evaluation',
              path_in_repo='evaluation', repo_id=os.environ['HF_REPO_ID'],
              token=os.environ['HF_TOKEN'])
"
```

**Training diverged again:** Check the golden sidecar log. The best checkpoint at `/workspace/_golden/best_ckpt/` is your recovery point. Load it as the final adapter and skip retraining.
