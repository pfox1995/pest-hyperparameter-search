#!/usr/bin/env python3
"""
Final Retrain — Pest Detection VLM (Qwen3.5-9B LoRA)
======================================================
Freezes the winning hyperparameters from W&B trial 015 and runs a single
full-data training on the RTX PRO 6000 Blackwell (96GB, sm_100).

Reuses the proven data pipeline from hp_search.py (image caching,
crop_tight augmentation, Unsloth collator, evaluation). No Optuna, no
search — one run, fixed config.

Usage (on /workspace after setup_runpod.sh installs deps + data):
    python3 train_final.py                        # Default: 2 epochs, auto-uploads to HF
    python3 train_final.py --epochs 3             # Push to 3 epochs
    python3 train_final.py --no-eval --no-wandb   # Minimal run, no HF upload

Required env vars for full functionality:
    WANDB_API_KEY          — enables W&B logging
    DISCORD_WEBHOOK_URL    — enables Discord notifications
    HF_TOKEN               — required for dataset download + HF Hub upload
    HF_REPO_ID             — target repo for HF upload (e.g. "user/pest-detector")
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

import torch

# ─── Import proven pipeline helpers from the main search script ───────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hp_search import (  # noqa: E402
    BASE_MODEL,
    BEST_MODEL_DIR,
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
# FROZEN HYPERPARAMETERS — W&B trial 015
# Keep these EXACTLY as-is. The LR was co-tuned with effective_batch=8;
# changing batch size requires re-tuning LR (roughly √-law scaling).
# ═══════════════════════════════════════════════════════════════════════

HP = {
    # LoRA
    "lora_r":             64,
    "lora_alpha":         128,          # = r * alpha_ratio(2)
    "use_rslora":         True,
    "lora_dropout":       0.0,
    "finetune_vision":    False,        # language layers only

    # Optimizer
    "learning_rate":      0.00011645105452323228,
    "weight_decay":       0.013802470048539942,
    "lr_scheduler_type":  "linear",
    "warmup_steps":       0,
    "max_grad_norm":      1.0,
    "adam_beta1":         0.9,
    "adam_beta2":         0.999,
    "adam_epsilon":       1e-8,
    "optim":              "adamw_8bit",

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

# Default output path matches hp_search.BEST_MODEL_DIR so tools pointing
# at that path (eval scripts, legacy glue) still find the adapter.
# Override via --output-dir or HP_BEST_MODEL_DIR env var.
FINAL_OUTPUT_DIR = os.environ.get("FINAL_OUTPUT_DIR", BEST_MODEL_DIR)


# ═══════════════════════════════════════════════════════════════════════
# MODEL CARD
# ═══════════════════════════════════════════════════════════════════════

def _build_readme(epochs: int, elapsed_min: float, eval_result: dict | None) -> str:
    """Minimal HuggingFace model card — HPs, training config, eval metrics."""
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
---

# Pest Detection VLM — Qwen3.5-9B LoRA

Fine-tuned LoRA adapter for 19-class Korean pest classification.
Trained with Unsloth on an RTX PRO 6000 Blackwell (96GB).

## Training config

| Hyperparameter | Value |
|---|---|
| LoRA rank | {HP['lora_r']} |
| LoRA alpha | {HP['lora_alpha']} |
| Use rsLoRA | {HP['use_rslora']} |
| Finetune vision layers | {HP['finetune_vision']} |
| Learning rate | {HP['learning_rate']:.6f} |
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

HPs selected via Optuna TPE search (trial 015 of 60 total).

{metrics_section}
## Usage

```python
from unsloth import FastVisionModel
model, tokenizer = FastVisionModel.from_pretrained(
    "{BASE_MODEL}",
    load_in_4bit=False,
)
model.load_adapter("<this-repo-id>")
FastVisionModel.for_inference(model)
```
"""


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Final retrain with frozen trial_015 HPs on PRO 6000",
    )
    parser.add_argument(
        "--epochs", type=int, default=2,
        help="Training epochs. Default 2 matches trial 015's sampled "
             "value and the VLM-classification literature sweet spot "
             "(Unsloth: 1-3 epochs; Qwen2.5-VL community: 3 common but "
             "epoch-2 often peaks). Override to 3 if you have budget.",
    )
    parser.add_argument(
        "--output-dir", default=FINAL_OUTPUT_DIR,
        help="Where to save the final LoRA adapter + eval results.",
    )
    parser.add_argument(
        "--save-strategy", default="epoch", choices=["no", "epoch", "steps"],
        help="Checkpoint strategy. 'epoch' saves after each epoch (safer "
             "for long runs; you can resume if pod dies).",
    )
    parser.add_argument(
        "--save-steps", type=int, default=500,
        help="Save every N steps (only if --save-strategy=steps).",
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Skip post-training evaluation (confusion matrix + F1).",
    )
    parser.add_argument(
        "--eval-samples", type=int, default=500,
        help="Validation samples for post-training eval.",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable W&B logging.",
    )
    parser.add_argument(
        "--hf-repo", default=os.environ.get("HF_REPO_ID", ""),
        help="HuggingFace repo to push final adapter + tokenizer + eval "
             "artifacts + README. Defaults to HF_REPO_ID env var. "
             "Repo created as private if it doesn't exist.",
    )
    args = parser.parse_args()

    # ─── Deterministic seeds ──────────────────────────────────────────
    random.seed(HP["seed"])
    torch.manual_seed(HP["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(HP["seed"])

    # ─── Verify GPU — expect sm_100 (Blackwell) ───────────────────────
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
            f"GPU has only {vram_gb:.0f}GB VRAM — trial 015 was tuned for "
            f"a 48GB card. You may OOM at batch=1, seq=1024."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Load data ────────────────────────────────────────────────────
    logger.info("데이터셋 로딩 중...")
    n_train_total = _get_line_count(os.path.join(DATA_DIR, "train.jsonl"))
    train_frac = get_max_data_fraction(n_train_total)
    if train_frac < 1.0:
        logger.info(
            f"RAM 제한으로 학습 데이터 {train_frac:.0%}만 사용 "
            f"({n_train_total}개 총 샘플 기준)"
        )

    train_dataset = load_dataset_from_jsonl(
        "train",
        tight_prob=HP["crop_tight_prob"],
        fraction=train_frac,
    )
    val_dataset = load_dataset_from_jsonl("val")
    logger.info(
        f"학습 샘플: {len(train_dataset)}, 검증 샘플: {len(val_dataset)}"
    )

    # ─── Load model + attach LoRA ─────────────────────────────────────
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

    # ─── W&B setup ────────────────────────────────────────────────────
    wandb_run = None
    report_to = "none"
    if not args.no_wandb and wandb_is_available():
        import wandb
        run_name = f"final-trial015-{datetime.now(KST).strftime('%m%d-%H%M')}"
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY or None,
            name=run_name,
            config={**HP, "num_train_epochs": args.epochs,
                    "source": "trial_015_frozen"},
            tags=["final", "trial-015", "pro-6000"],
            reinit=True,
        )
        report_to = "wandb"
        logger.info(f"W&B 실행: {run_name}")

    # ─── Trainer ──────────────────────────────────────────────────────
    sft_args = SFTConfig(
        # Batching
        per_device_train_batch_size=HP["per_device_train_batch_size"],
        gradient_accumulation_steps=HP["gradient_accumulation_steps"],
        per_device_eval_batch_size=4,

        # Duration
        num_train_epochs=args.epochs,

        # Optimizer
        learning_rate=HP["learning_rate"],
        weight_decay=HP["weight_decay"],
        lr_scheduler_type=HP["lr_scheduler_type"],
        warmup_steps=HP["warmup_steps"],
        max_grad_norm=HP["max_grad_norm"],
        adam_beta1=HP["adam_beta1"],
        adam_beta2=HP["adam_beta2"],
        adam_epsilon=HP["adam_epsilon"],
        optim=HP["optim"],

        # Precision (Blackwell bf16 is fast; fp8 not used — Unsloth/TRL
        # doesn't auto-enable it for Qwen-VL training)
        bf16=True,
        bf16_full_eval=True,
        fp16=False,

        # Memory
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,

        # Checkpointing + eval
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="epoch" if not args.no_eval else "no",
        eval_accumulation_steps=2,

        # Load best at end only if we're saving epoch checkpoints
        load_best_model_at_end=(args.save_strategy == "epoch" and not args.no_eval),
        metric_for_best_model="eval_loss" if not args.no_eval else None,
        greater_is_better=False,

        # Misc
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
        f"🚀 **최종 학습 시작** (PRO 6000)\n"
        f"• 에폭: {args.epochs}\n"
        f"• LR: {HP['learning_rate']:.2e}\n"
        f"• LoRA r={HP['lora_r']}, α={HP['lora_alpha']}\n"
        f"• 학습 샘플: {len(train_dataset)}"
    )
    t0 = time.time()
    train_result = trainer.train()
    elapsed_min = (time.time() - t0) / 60
    logger.info(f"학습 완료. 소요 시간: {elapsed_min:.1f}분")

    # ─── Save LoRA adapter + tokenizer ────────────────────────────────
    lora_dir = os.path.join(args.output_dir, "lora")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    logger.info(f"LoRA 어댑터 저장: {lora_dir}")

    # ─── Post-training evaluation ─────────────────────────────────────
    eval_result = None
    if not args.no_eval:
        logger.info("최종 평가 실행 중...")
        FastVisionModel.for_inference(model)
        eval_dir = os.path.join(args.output_dir, "evaluation")
        eval_result = evaluate_model(
            model, tokenizer, val_dataset,
            max_samples=args.eval_samples,
            save_dir=eval_dir,
            trial_num="final",
        )
        logger.info(
            f"최종 평가:\n"
            f"  정확도:         {eval_result['accuracy']:.4f}\n"
            f"  F1 (macro):     {eval_result['f1_macro']:.4f}\n"
            f"  F1 (weighted):  {eval_result['f1_weighted']:.4f}\n"
            f"  Precision:      {eval_result['precision_macro']:.4f}\n"
            f"  Recall:         {eval_result['recall_macro']:.4f}"
        )

        import json
        with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
            json.dump(eval_result, f, indent=2, default=str, ensure_ascii=False)

        if wandb_run:
            import wandb
            wandb.log({
                "final/accuracy":      eval_result["accuracy"],
                "final/f1_macro":      eval_result["f1_macro"],
                "final/f1_weighted":   eval_result["f1_weighted"],
                "final/precision":     eval_result["precision_macro"],
                "final/recall":        eval_result["recall_macro"],
                "final/train_minutes": elapsed_min,
            })

    # ─── Discord notification ─────────────────────────────────────────
    if eval_result:
        discord_send(
            f"✅ **최종 학습 완료** ({elapsed_min:.0f}분)\n"
            f"• 정확도: {eval_result['accuracy']:.4f}\n"
            f"• F1 (macro): {eval_result['f1_macro']:.4f}\n"
            f"• 모델: `{lora_dir}`"
        )
    else:
        discord_send(f"✅ **최종 학습 완료** ({elapsed_min:.0f}분)")

    # ─── HuggingFace Hub upload (adapter + tokenizer + eval artifacts) ─
    hf_repo = args.hf_repo or os.environ.get("HF_REPO_ID", "")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_repo and hf_token:
        try:
            logger.info(f"HuggingFace 업로드 시작: {hf_repo}")

            # 1. Push LoRA adapter (creates repo if needed)
            model.push_to_hub(hf_repo, token=hf_token, private=True)
            tokenizer.push_to_hub(hf_repo, token=hf_token, private=True)

            # 2. Upload evaluation artifacts (confusion matrix, metrics.json)
            eval_dir = os.path.join(args.output_dir, "evaluation")
            if os.path.isdir(eval_dir):
                from huggingface_hub import upload_folder
                upload_folder(
                    folder_path=eval_dir,
                    path_in_repo="evaluation",
                    repo_id=hf_repo,
                    token=hf_token,
                    commit_message="Add final evaluation metrics + confusion matrix",
                )
                logger.info("  + 평가 아티팩트 업로드 완료")

            # 3. Write a minimal README with HPs + metrics
            try:
                from huggingface_hub import upload_file
                readme = _build_readme(args.epochs, elapsed_min, eval_result)
                readme_path = os.path.join(args.output_dir, "README.md")
                with open(readme_path, "w", encoding="utf-8") as f:
                    f.write(readme)
                upload_file(
                    path_or_fileobj=readme_path,
                    path_in_repo="README.md",
                    repo_id=hf_repo,
                    token=hf_token,
                    commit_message="Add model card",
                )
                logger.info("  + 모델 카드 (README.md) 업로드 완료")
            except Exception as e:
                logger.warning(f"README 업로드 실패: {e}")

            hf_url = f"https://huggingface.co/{hf_repo}"
            logger.info(f"HuggingFace 업로드 완료: {hf_url}")
            discord_send(f"🤗 HuggingFace: {hf_url}")

        except Exception as e:
            logger.warning(f"HuggingFace 업로드 실패: {e}")
            discord_send(f"⚠️ HuggingFace 업로드 실패: {e}")
    else:
        missing = []
        if not hf_repo:
            missing.append("HF_REPO_ID (또는 --hf-repo)")
        if not hf_token:
            missing.append("HF_TOKEN")
        logger.warning(
            f"HuggingFace 업로드 건너뜀 — 미설정: {', '.join(missing)}"
        )

    if wandb_run:
        wandb_run.finish()

    logger.info(f"✅ 모든 작업 완료. 모델: {lora_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
