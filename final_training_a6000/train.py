#!/usr/bin/env python3
"""
Final Retrain — Pest Detection VLM on RTX A6000 (48 GB, Ampere sm_86)
======================================================================
One-shot training using frozen hyperparameters from Optuna trial 015,
hardened against the divergence observed in W&B run zhgoxqpx.

Differences vs the parent train_final.py:
  * Lives in its own folder; imports dataset/eval helpers from the
    parent hp_search.py via sys.path insertion. Keeps shared logic
    DRY while staying self-contained for A6000 deploys.
  * A6000 is a 48 GB Ampere card — bf16 is native but VRAM is tighter
    than the 96 GB Blackwell this was first run on. Per-device eval
    batch size is reduced to 2 (from 4) as insurance.
  * --eval-samples defaults to the full val set (teacher feedback:
    500 samples over 19 classes was too thin for reliable per-class
    F1; full 1595 gives ~84/class).
  * adamw_torch + warmup_ratio=0.03 + save_steps=25 baked in (not
    flags) — these are stability requirements, not hyperparameters.

Usage on the pod (after bash setup_a6000.sh completes):
    bash run.sh                      # 1 epoch, full val eval
    EPOCHS=2 bash run.sh             # override epoch count
    EXTRA_ARGS='--no-wandb' bash run.sh
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

import torch

# ─── Import shared helpers from parent hp_search.py ────────────────────
# Parent is the repo root; this script is in final_training_a6000/
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PARENT)

from hp_search import (  # noqa: E402
    BASE_MODEL,
    DATA_DIR,
    RANDOM_SEED,
    VOLUME_DIR,
    WANDB_ENTITY,
    WANDB_PROJECT,
    _get_line_count,
    clear_gpu_memory,
    discord_send,
    evaluate_model,
    get_max_data_fraction,
    load_dataset_from_jsonl,
    load_model_with_retry,
    logger,
    wandb_is_available,
)

KST = timezone(timedelta(hours=9))

# ═══════════════════════════════════════════════════════════════════════
# FROZEN HYPERPARAMETERS — Optuna trial 015
# Unchanged from train_final.py; these are the tuned values. All A6000
# adjustments happen in argparse defaults / SFTConfig wiring below.
# ═══════════════════════════════════════════════════════════════════════

HP = {
    # LoRA
    "lora_r":             64,
    "lora_alpha":         128,
    "use_rslora":         True,
    "lora_dropout":       0.0,
    "finetune_vision":    False,

    # Optimizer — adamw_torch + warmup_ratio are stability requirements
    # (adamw_8bit + warmup_steps=0 caused the divergence in run zhgoxqpx)
    "learning_rate":      0.00011645105452323228,
    "weight_decay":       0.013802470048539942,
    "lr_scheduler_type":  "linear",
    "warmup_ratio":       0.03,
    "max_grad_norm":      1.0,
    "adam_beta1":         0.9,
    "adam_beta2":         0.999,
    "adam_epsilon":       1e-8,
    "optim":              "adamw_torch",

    # Batch / sequence
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,   # effective batch = 8
    "max_seq_length":     1024,

    # Data augmentation
    "crop_tight_prob":    0.4560998466814129,

    # Seeds
    "seed":               42,
    "data_seed":          3407,
}

DEFAULT_OUTPUT_DIR = os.environ.get(
    "FINAL_OUTPUT_DIR", "/workspace/best-pest-detector"
)


# ═══════════════════════════════════════════════════════════════════════
# Model card
# ═══════════════════════════════════════════════════════════════════════

def build_readme(epochs: int, elapsed_min: float, eval_result: dict | None) -> str:
    metrics_section = ""
    if eval_result:
        metrics_section = f"""## Evaluation

| Metric | Value |
|---|---|
| Accuracy | {eval_result['accuracy']:.4f} |
| F1 (macro) | {eval_result['f1_macro']:.4f} |
| F1 (weighted) | {eval_result['f1_weighted']:.4f} |
| Precision (macro) | {eval_result['precision_macro']:.4f} |
| Recall (macro) | {eval_result['recall_macro']:.4f} |
| Val samples | {eval_result.get('total', '?')} |

Confusion matrix and per-class metrics in `evaluation/`.
"""
    return f"""---
base_model: {BASE_MODEL}
library_name: peft
tags:
- lora
- vision-language
- pest-detection
- unsloth
- rtx-a6000
---

# Pest Detection VLM — Qwen3.5-9B LoRA (A6000 retrain)

Fine-tuned LoRA adapter for 19-class Korean pest classification.
Trained with Unsloth on an RTX A6000 (48 GB, Ampere).

## Training config

| Hyperparameter | Value |
|---|---|
| LoRA rank | {HP['lora_r']} |
| LoRA alpha | {HP['lora_alpha']} |
| Use rsLoRA | {HP['use_rslora']} |
| Finetune vision layers | {HP['finetune_vision']} |
| Learning rate | {HP['learning_rate']:.6f} |
| Warmup ratio | {HP['warmup_ratio']} |
| Weight decay | {HP['weight_decay']:.6f} |
| LR scheduler | {HP['lr_scheduler_type']} |
| Optimizer | {HP['optim']} |
| Per-device batch | {HP['per_device_train_batch_size']} |
| Grad accumulation | {HP['gradient_accumulation_steps']} |
| Effective batch | {HP['per_device_train_batch_size'] * HP['gradient_accumulation_steps']} |
| Max seq length | {HP['max_seq_length']} |
| Epochs | {epochs} |
| Crop-tight prob | {HP['crop_tight_prob']:.4f} |
| Precision | bf16 |
| Gradient checkpointing | True |
| Training time | {elapsed_min:.0f} min |

HPs from Optuna TPE search (trial 015). Stability hardening applied
(adamw_torch, warmup_ratio=0.03) after run zhgoxqpx diverged.

{metrics_section}
## Usage

```python
from unsloth import FastVisionModel
model, tokenizer = FastVisionModel.from_pretrained("{BASE_MODEL}")
model.load_adapter("<this-repo-id>")
FastVisionModel.for_inference(model)
```
"""


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Final retrain with trial_015 HPs on RTX A6000",
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Training epochs. Default 1: convergence happens at step "
             "~21 (~1.5%% of epoch 1) with this HP set. 1 epoch + "
             "frequent checkpointing + load_best_model_at_end is "
             "sufficient. Use 2 only if you have budget and want a "
             "second pass for robustness.",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Where to save the final LoRA adapter + eval results.",
    )
    parser.add_argument(
        "--save-steps", type=int, default=25,
        help="Save a checkpoint every N steps. Default 25 ≈ every "
             "3-4 min on A6000; small enough to catch the "
             "pre-divergence window if optimizer misbehaves.",
    )
    parser.add_argument(
        "--eval-samples", type=int, default=-1,
        help="Validation samples for end-of-training CM. Default -1 = "
             "use the entire val set (~1595). Small values produce "
             "unreliable per-class F1 on 19-class tasks.",
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Skip post-training evaluation (CM + F1).",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable W&B logging.",
    )
    parser.add_argument(
        "--hf-repo", default=os.environ.get("HF_REPO_ID", ""),
        help="HuggingFace repo to push final adapter + tokenizer + "
             "eval artifacts. Defaults to HF_REPO_ID env var. "
             "Repo created as private if it doesn't exist.",
    )
    args = parser.parse_args()

    # ─── Deterministic seeds ──────────────────────────────────────────
    random.seed(HP["seed"])
    torch.manual_seed(HP["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(HP["seed"])

    # ─── GPU sanity check (expect A6000 / Ampere) ─────────────────────
    assert torch.cuda.is_available(), "CUDA GPU required"
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    capability = torch.cuda.get_device_capability(0)
    logger.info(
        f"GPU: {gpu_name} | VRAM: {vram_gb:.0f}GB | "
        f"compute_capability: sm_{capability[0]}{capability[1]}"
    )
    if vram_gb < 40:
        logger.warning(
            f"GPU has only {vram_gb:.0f}GB VRAM — trial 015 was tuned "
            "for a 48GB card. You may OOM at batch=1, seq=1024."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Data ─────────────────────────────────────────────────────────
    logger.info("데이터셋 로딩 중...")
    n_train_total = _get_line_count(os.path.join(DATA_DIR, "train.jsonl"))
    train_frac = get_max_data_fraction(n_train_total)
    if train_frac < 1.0:
        logger.info(
            f"RAM 제한으로 학습 데이터 {train_frac:.0%}만 사용 "
            f"({n_train_total}개 총 샘플 기준)"
        )

    # HP_LAZY_DATASET=1 (set by run.sh) returns LazyImageDataset —
    # decodes per-__getitem__ so we skip the ~20 GB preload.
    train_dataset = load_dataset_from_jsonl(
        "train",
        tight_prob=HP["crop_tight_prob"],
        fraction=train_frac,
    )
    val_dataset = load_dataset_from_jsonl("val")
    n_val = len(val_dataset)
    logger.info(f"학습 샘플: {len(train_dataset)}, 검증 샘플: {n_val}")

    eval_samples = n_val if args.eval_samples < 0 else min(args.eval_samples, n_val)

    # ─── Model + LoRA ─────────────────────────────────────────────────
    clear_gpu_memory()
    logger.info(f"모델 로딩: {BASE_MODEL}")
    model, tokenizer = load_model_with_retry(BASE_MODEL)

    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=HP["finetune_vision"],
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=HP["lora_r"],
        lora_alpha=HP["lora_alpha"],
        lora_dropout=HP["lora_dropout"],
        bias="none",
        random_state=HP["data_seed"],
        use_rslora=HP["use_rslora"],
    )
    FastVisionModel.for_training(model)

    # ─── W&B ──────────────────────────────────────────────────────────
    wandb_run = None
    report_to = "none"
    if not args.no_wandb and wandb_is_available():
        import wandb
        run_name = f"a6000-final-{datetime.now(KST).strftime('%m%d-%H%M')}"
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY or None,
            name=run_name,
            config={**HP, "num_train_epochs": args.epochs,
                    "gpu": "rtx_a6000", "source": "trial_015_frozen"},
            tags=["final", "trial-015", "a6000"],
            reinit=True,
        )
        report_to = "wandb"
        logger.info(f"W&B 실행: {run_name}")

    # ─── SFTConfig — stability fixes baked in ─────────────────────────
    sft_args = SFTConfig(
        per_device_train_batch_size=HP["per_device_train_batch_size"],
        gradient_accumulation_steps=HP["gradient_accumulation_steps"],
        per_device_eval_batch_size=2,  # A6000: halved from 4 for VRAM headroom
        num_train_epochs=args.epochs,

        # Optimizer (stability hardening)
        learning_rate=HP["learning_rate"],
        weight_decay=HP["weight_decay"],
        lr_scheduler_type=HP["lr_scheduler_type"],
        warmup_ratio=HP["warmup_ratio"],
        max_grad_norm=HP["max_grad_norm"],
        adam_beta1=HP["adam_beta1"],
        adam_beta2=HP["adam_beta2"],
        adam_epsilon=HP["adam_epsilon"],
        optim=HP["optim"],

        # Precision
        bf16=True,
        bf16_full_eval=True,
        fp16=False,

        # Memory
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,

        # Save + eval cadence (must match for load_best_model_at_end)
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy=("steps" if not args.no_eval else "no"),
        eval_steps=args.save_steps,
        eval_accumulation_steps=2,

        load_best_model_at_end=(not args.no_eval),
        metric_for_best_model="eval_loss" if not args.no_eval else None,
        greater_is_better=False,

        logging_steps=20,
        logging_strategy="steps",
        seed=HP["seed"],
        data_seed=HP["data_seed"],
        output_dir=args.output_dir,
        report_to=report_to,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=HP["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset if not args.no_eval else None,
        args=sft_args,
    )

    # ─── Train ────────────────────────────────────────────────────────
    discord_send(
        f"🚀 **A6000 최종 학습 시작**\n"
        f"• 에폭: {args.epochs}\n"
        f"• LR: {HP['learning_rate']:.2e} (warmup_ratio={HP['warmup_ratio']})\n"
        f"• LoRA r={HP['lora_r']}, α={HP['lora_alpha']}\n"
        f"• 학습 샘플: {len(train_dataset)}\n"
        f"• 검증 샘플: {n_val}\n"
        f"• 평가 샘플 수: {eval_samples}"
    )

    t0 = time.time()
    train_result = trainer.train()
    elapsed_min = (time.time() - t0) / 60
    logger.info(f"학습 완료. 소요 시간: {elapsed_min:.1f}분")

    # ─── Save LoRA + tokenizer ────────────────────────────────────────
    lora_dir = os.path.join(args.output_dir, "lora")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    logger.info(f"LoRA 어댑터 저장: {lora_dir}")

    # ─── End-of-training CM evaluation ────────────────────────────────
    eval_result = None
    if not args.no_eval:
        logger.info(
            f"최종 평가 실행 중 ({eval_samples}개 샘플, "
            "generation-based)..."
        )
        FastVisionModel.for_inference(model)
        eval_dir = os.path.join(args.output_dir, "evaluation")
        eval_result = evaluate_model(
            model, tokenizer, val_dataset,
            max_samples=eval_samples,
            save_dir=eval_dir,
            trial_num="a6000-final",
        )
        logger.info(
            f"최종 평가:\n"
            f"  정확도:         {eval_result['accuracy']:.4f}\n"
            f"  F1 (macro):     {eval_result['f1_macro']:.4f}\n"
            f"  F1 (weighted):  {eval_result['f1_weighted']:.4f}\n"
            f"  Precision:      {eval_result['precision_macro']:.4f}\n"
            f"  Recall:         {eval_result['recall_macro']:.4f}\n"
            f"  샘플 수:        {eval_result.get('total', '?')}"
        )

        import json
        with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
            json.dump(eval_result, f, indent=2, default=str, ensure_ascii=False)

    # ─── README + HF Hub upload ───────────────────────────────────────
    readme_path = os.path.join(args.output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(build_readme(args.epochs, elapsed_min, eval_result))

    hf_repo = args.hf_repo or os.environ.get("HF_REPO_ID", "")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_repo and hf_token:
        try:
            logger.info(f"HuggingFace 업로드 시작: {hf_repo}")
            model.push_to_hub(hf_repo, token=hf_token, private=True)
            tokenizer.push_to_hub(hf_repo, token=hf_token, private=True)
            from huggingface_hub import upload_folder, upload_file
            eval_dir = os.path.join(args.output_dir, "evaluation")
            if os.path.isdir(eval_dir):
                upload_folder(
                    folder_path=eval_dir,
                    path_in_repo="evaluation",
                    repo_id=hf_repo,
                    token=hf_token,
                )
            upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=hf_repo,
                token=hf_token,
            )
            hf_url = f"https://huggingface.co/{hf_repo}"
            logger.info(f"HuggingFace 업로드 완료: {hf_url}")
            discord_send(
                f"✅ **A6000 학습 완료 + 업로드**\n"
                f"• 시간: {elapsed_min:.0f}분\n"
                f"• eval_loss (최선): {trainer.state.best_metric}\n"
                f"• HF: {hf_url}"
            )
        except Exception as e:
            logger.exception(f"HF 업로드 실패: {e}")
            discord_send(
                f"⚠️ **A6000 학습 완료 — HF 업로드 실패**\n"
                f"LoRA는 {lora_dir} 에 저장됨.\n오류: {e}"
            )
    else:
        missing = []
        if not hf_repo:
            missing.append("HF_REPO_ID (또는 --hf-repo)")
        if not hf_token:
            missing.append("HF_TOKEN")
        logger.warning(
            f"HF 업로드 생략 — 누락된 환경변수: {', '.join(missing)}"
        )

    if wandb_run:
        wandb_run.finish()

    logger.info(f"✅ 모든 작업 완료. 모델: {lora_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
