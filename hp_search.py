#!/usr/bin/env python3
"""
Hyperparameter Search for Pest Detection Vision Model (Qwen3.5-9B LoRA)
========================================================================
Optuna + Unsloth + SFTTrainer + W&B + Discord Notifications

Usage:
    bash setup_runpod.sh                       # Full automated pipeline
    python hp_search.py --proxy --n-trials 50  # Phase 1: fast screening
    python hp_search.py --n-trials 10 --retrain # Phase 2: full + retrain
    python hp_search.py --analyze              # Analyze saved results

Environment Variables (set in setup_runpod.sh):
    WANDB_API_KEY          — Weights & Biases API key
    DISCORD_WEBHOOK_URL    — Discord webhook for Korean notifications
    GITHUB_TOKEN           — GitHub PAT (if needed)
    HF_TOKEN               — HuggingFace token (if dataset is private)
    HP_DATA_DIR            — Dataset path (default: /workspace/data)
    HP_VOLUME_DIR          — Persistent volume (default: /runpod-volume)
    HP_DB_PATH             — Optuna SQLite DB path
    HP_OUTPUT_DIR          — Trial outputs directory
    HP_BEST_MODEL_DIR      — Where to save the best model
    WANDB_PROJECT          — W&B project name
    WANDB_ENTITY           — W&B team/username
"""

import argparse
import gc
import json
import logging
import os
import random
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

import torch
from PIL import Image

# ═══════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

DATA_DIR       = os.environ.get("HP_DATA_DIR", "/workspace/data")
VOLUME_DIR     = os.environ.get("HP_VOLUME_DIR", "/workspace")
OUTPUT_BASE    = os.environ.get("HP_OUTPUT_DIR", f"{VOLUME_DIR}/hp_search_outputs")
BEST_MODEL_DIR = os.environ.get("HP_BEST_MODEL_DIR", f"{VOLUME_DIR}/best-pest-detector")
DB_PATH        = os.environ.get("HP_DB_PATH", f"{VOLUME_DIR}/hp_search_results.db")
STORAGE_DB     = f"sqlite:///{DB_PATH}"
LOG_FILE       = os.environ.get("HP_LOG_FILE", f"{VOLUME_DIR}/hp_search.log")

STUDY_NAME       = "pest-detection-hpsearch"
BASE_MODEL       = "unsloth/Qwen3.5-9B"
RANDOM_SEED      = 42
N_TRIALS_DEFAULT = 30

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
WANDB_PROJECT       = os.environ.get("WANDB_PROJECT", "pest-detection-hpsearch")
WANDB_ENTITY        = os.environ.get("WANDB_ENTITY", "")

QUICK_DATA_FRACTION = 0.2
QUICK_EPOCHS        = 1
PROXY_DATA_FRACTION = 0.1
PROXY_EPOCHS        = 1

SEARCH_SPACE = {
    "per_device_train_batch_size": [1, 2, 4],
    "gradient_accumulation_steps": [2, 4, 8],
    "learning_rate":              (1e-6, 5e-4),
    "num_train_epochs":           [2, 3, 5],
    "warmup_steps":               [0, 10, 50, 100],
    "weight_decay":               (0.0, 0.1),
    "lr_scheduler_type":          ["linear", "cosine", "cosine_with_restarts"],
    "max_seq_length":             [1024, 2048],
    "lora_r":                     [4, 8, 16, 32, 64],
    "lora_alpha_ratio":           [0.5, 1.0, 2.0, 4.0],
    "finetune_vision_layers":     [True, False],
    "use_rslora":                 [False, True],
    # 50/50 tight crop vs original (no crop)
    # crop_tight_prob = fraction using tight bbox crop; rest uses original
    "crop_tight_prob":            (0.3, 0.7),
}

SYSTEM_MSG = (
    "당신은 작물 해충 식별 전문가입니다. "
    "사진을 보고 해충의 이름만 한국어로 답하세요. "
    '해충이 없으면 "정상"이라고만 답하세요. '
    "부가 설명 없이 이름만 출력하세요."
)

PROMPTS = [
    "이 사진에 있는 해충의 이름을 알려주세요.",
    "이 벌레는 무엇인가요?",
    "사진 속 해충을 식별해주세요.",
    "이 작물에 있는 해충의 종류가 무엇인가요?",
    "이 사진에서 어떤 해충이 보이나요?",
]

OBJECTIVE_METRIC = "eval_loss"

# Logger is set up in main() to avoid crashing on import when volume is absent
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 2. DISCORD NOTIFICATIONS (Korean, async)
# ═══════════════════════════════════════════════════════════════════════

KST = timezone(timedelta(hours=9))

def _kst_now():
    return datetime.now(KST).strftime("%m/%d %H:%M")

def discord_send(content: str = None, embed: dict = None):
    """Fire-and-forget Discord message via background thread."""
    if not DISCORD_WEBHOOK_URL:
        return
    def _send():
        import requests as _req
        payload = {}
        if content:
            payload["content"] = content
        if embed:
            payload["embeds"] = [embed]
        try:
            _req.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        except Exception as e:
            logger.warning(f"Discord 알림 실패: {e}")
    threading.Thread(target=_send, daemon=True).start()

def discord_search_started(n_trials, mode):
    discord_send(embed={
        "title": "🔍 하이퍼파라미터 검색 시작",
        "color": 0x3498db,
        "fields": [
            {"name": "모드", "value": mode, "inline": True},
            {"name": "트라이얼 수", "value": str(n_trials), "inline": True},
            {"name": "모델", "value": BASE_MODEL, "inline": True},
            {"name": "목적 함수", "value": OBJECTIVE_METRIC, "inline": True},
        ],
        "footer": {"text": f"시작: {_kst_now()} KST"},
    })

def discord_trial_complete(trial_num, value, params, duration_min,
                           is_best, completed, total):
    color = 0x2ecc71 if is_best else 0x95a5a6
    title = (f"🏆 트라이얼 #{trial_num} — 새로운 최고 기록!"
             if is_best else f"✅ 트라이얼 #{trial_num} 완료")
    bs = params.get("batch_size", 0) * params.get("grad_accum", 0)
    discord_send(embed={
        "title": title, "color": color,
        "fields": [
            {"name": "Eval Loss", "value": f"`{value:.6f}`", "inline": True},
            {"name": "소요 시간", "value": f"{duration_min:.1f}분", "inline": True},
            {"name": "진행률", "value": f"{completed}/{total}", "inline": True},
            {"name": "학습률", "value": f"`{params.get('learning_rate',0):.2e}`", "inline": True},
            {"name": "LoRA r", "value": str(params.get("lora_r", "?")), "inline": True},
            {"name": "배치 크기", "value": str(bs), "inline": True},
        ],
        "footer": {"text": f"{_kst_now()} KST"},
    })

def discord_trial_pruned(trial_num, reason):
    discord_send(embed={
        "title": f"✂️ 트라이얼 #{trial_num} 가지치기",
        "description": reason, "color": 0xe67e22,
        "footer": {"text": f"{_kst_now()} KST"},
    })

def discord_trial_error(trial_num, error):
    discord_send(embed={
        "title": f"❌ 트라이얼 #{trial_num} 오류",
        "description": f"```{str(error)[:500]}```", "color": 0xe74c3c,
        "footer": {"text": f"{_kst_now()} KST"},
    })

def discord_phase_complete(phase, best_value, best_params,
                           completed, pruned, total_time_hr):
    r = best_params.get("lora_r", "?")
    alpha_ratio = best_params.get("lora_alpha_ratio", 1)
    alpha = int(r * alpha_ratio) if isinstance(r, int) else "?"
    discord_send(embed={
        "title": f"🎯 {phase} 완료!",
        "color": 0x9b59b6,
        "fields": [
            {"name": "최적 Eval Loss", "value": f"`{best_value:.6f}`", "inline": True},
            {"name": "총 소요 시간", "value": f"{total_time_hr:.1f}시간", "inline": True},
            {"name": "완료/가지치기", "value": f"{completed}/{pruned}", "inline": True},
            {"name": "최적 학습률", "value": f"`{best_params.get('learning_rate',0):.2e}`", "inline": True},
            {"name": "최적 LoRA", "value": f"r={r}, a={alpha}", "inline": True},
            {"name": "비전 레이어", "value": "O" if best_params.get("finetune_vision") else "X", "inline": True},
        ],
        "footer": {"text": f"완료: {_kst_now()} KST"},
    })

def discord_retrain_complete(accuracy, model_path):
    discord_send(embed={
        "title": "🚀 최적 모델 학습 완료!",
        "description": f"최종 정확도: **{accuracy:.2%}**\n모델 경로: `{model_path}`",
        "color": 0x2ecc71,
        "footer": {"text": f"완료: {_kst_now()} KST"},
    })

def discord_error(error):
    discord_send(embed={
        "title": "💥 치명적 오류 발생",
        "description": f"```{str(error)[:800]}```", "color": 0xe74c3c,
        "footer": {"text": f"{_kst_now()} KST"},
    })


# ═══════════════════════════════════════════════════════════════════════
# 3. W&B INTEGRATION
# ═══════════════════════════════════════════════════════════════════════

def wandb_is_available():
    return bool(os.environ.get("WANDB_API_KEY"))

def wandb_init_trial(trial_num, params):
    if not wandb_is_available():
        return None
    import wandb
    return wandb.init(
        project=WANDB_PROJECT, entity=WANDB_ENTITY or None,
        name=f"trial-{trial_num:03d}", group=STUDY_NAME,
        config=params, reinit=True,
        tags=["hp-search", "pest-detection", "qwen3.5-9b"],
    )

def wandb_finish(run, exit_code=0):
    """Safely finish a W&B run. No-op if run is None."""
    if run is None:
        return
    try:
        import wandb
        wandb.finish(exit_code=exit_code)
    except Exception:
        pass

def wandb_log_best_summary(study):
    if not wandb_is_available():
        return
    import wandb
    best = study.best_trial
    run = wandb.init(
        project=WANDB_PROJECT, entity=WANDB_ENTITY or None,
        name="best-params-summary", group=STUDY_NAME,
        reinit=True, tags=["summary"],
    )
    wandb.config.update(best.params)
    wandb.log({
        "best_eval_loss": best.value,
        "best_train_loss": best.user_attrs.get("train_loss", 0),
        "best_trial_number": best.number,
    })
    wandb.finish()


# ═══════════════════════════════════════════════════════════════════════
# 4. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def crop_to_bbox(img, bbox, padding_ratio=0.0):
    xtl, ytl = bbox["xtl"], bbox["ytl"]
    xbr, ybr = bbox["xbr"], bbox["ybr"]
    bw, bh = xbr - xtl, ybr - ytl
    pad_x, pad_y = int(bw * padding_ratio), int(bh * padding_ratio)
    x1 = max(0, xtl - pad_x)
    y1 = max(0, ytl - pad_y)
    x2 = min(img.width, xbr + pad_x)
    y2 = min(img.height, ybr + pad_y)
    return img.crop((x1, y1, x2, y2))


def find_label_json(split, class_name, img_filename):
    json_path = os.path.join(DATA_DIR, split, class_name, img_filename + ".json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    for obj in data["annotations"]["object"]:
        if obj["grow"] == 33 and obj.get("points"):
            return obj["points"][0]
    return None


def make_conversation(image, label):
    return {
        "messages": [
            {"role": "system", "content": [
                {"type": "text", "text": SYSTEM_MSG}
            ]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": random.choice(PROMPTS)}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": label}
            ]},
        ]
    }


def load_dataset_from_jsonl(split, tight_prob=0.5, fraction=1.0):
    """Load dataset with 2-way crop strategy for pest images.

    For pest images with bounding boxes:
      - Tight crop (bbox only):    probability = tight_prob
      - Original image (no crop):  probability = 1 - tight_prob

    Normal ("정상") images always use the original.
    """
    jsonl_path = os.path.join(DATA_DIR, f"{split}.jsonl")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    if fraction < 1.0:
        keep = set(random.sample(range(total_lines), int(total_lines * fraction)))
    else:
        keep = None

    dataset = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if keep is not None and i not in keep:
                continue

            record = json.loads(line)
            messages = record["messages"]
            label = messages[-1]["content"][0]["text"]

            img_rel_path = None
            for msg in messages:
                for content in msg["content"]:
                    if content["type"] == "image" and "image" in content:
                        img_rel_path = content["image"]
                        break

            if img_rel_path is None:
                continue

            parts = img_rel_path.replace("\\", "/").split("/")
            if len(parts) < 3:
                logger.warning(f"예상치 못한 이미지 경로: {img_rel_path}, 건너뜀")
                continue

            img_path = os.path.join(DATA_DIR, img_rel_path)
            if not os.path.exists(img_path):
                continue

            class_name, img_filename = parts[1], parts[2]

            if label == "정상":
                img = Image.open(img_path).convert("RGB")
                dataset.append(make_conversation(img, label))
            else:
                bbox = find_label_json(split, class_name, img_filename)
                if bbox:
                    img = Image.open(img_path).convert("RGB")
                    if random.random() < tight_prob:
                        # Tight crop — pest morphology
                        cropped = crop_to_bbox(img, bbox, padding_ratio=0.0)
                        img.close()
                        dataset.append(make_conversation(cropped, label))
                    else:
                        # Original image — learn to locate pest in full frame
                        dataset.append(make_conversation(img, label))
                else:
                    img = Image.open(img_path).convert("RGB")
                    dataset.append(make_conversation(img, label))

    random.shuffle(dataset)
    return dataset


# ═══════════════════════════════════════════════════════════════════════
# 5. GPU MEMORY MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(
            f"GPU 메모리 — allocated: {allocated:.1f}GB, "
            f"reserved: {reserved:.1f}GB | peak 초기화됨"
        )


# ═══════════════════════════════════════════════════════════════════════
# 6. EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate_accuracy(model, tokenizer, val_dataset, max_samples=200):
    from collections import Counter

    samples = val_dataset[:max_samples]
    correct, total = 0, 0
    per_class_correct = Counter()
    per_class_total = Counter()
    misclassifications = []

    for item in samples:
        messages = item["messages"]
        ground_truth = messages[-1]["content"][0]["text"]

        # Image is stored as PIL object in the conversation dict
        image = messages[1]["content"][0]["image"]

        infer_messages = [
            {"role": "system", "content": [
                {"type": "text", "text": SYSTEM_MSG}
            ]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": messages[1]["content"][1]["text"]}
            ]}
        ]

        try:
            input_text = tokenizer.apply_chat_template(
                infer_messages, add_generation_prompt=True
            )
            inputs = tokenizer(
                image, input_text,
                add_special_tokens=False, return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=20,
                    temperature=0.1, use_cache=True,
                )

            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            per_class_total[ground_truth] += 1
            total += 1

            if generated == ground_truth:
                correct += 1
                per_class_correct[ground_truth] += 1
            else:
                misclassifications.append({
                    "truth": ground_truth, "predicted": generated,
                })

        except Exception as e:
            logger.warning(f"추론 오류: {e}")
            continue

    accuracy = correct / total if total > 0 else 0.0
    per_class_acc = {
        cls: per_class_correct.get(cls, 0) / per_class_total[cls]
        for cls in per_class_total if per_class_total[cls] > 0
    }

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_class_accuracy": per_class_acc,
        "top_misclassifications": misclassifications[:20],
    }


# ═══════════════════════════════════════════════════════════════════════
# 7. OBJECTIVE FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def objective(trial: optuna.Trial, args) -> float:
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback

    trial_start = time.time()
    trial_dir = os.path.join(OUTPUT_BASE, f"trial_{trial.number:03d}")
    os.makedirs(trial_dir, exist_ok=True)

    # Initialize as None for safe cleanup in finally block
    model, tokenizer, trainer = None, None, None
    wandb_run = None
    train_dataset, val_dataset = None, None

    try:
        # ─── Sample hyperparameters ───────────────────────────────────

        batch_size = trial.suggest_categorical("batch_size", SEARCH_SPACE["per_device_train_batch_size"])
        grad_accum = trial.suggest_categorical("grad_accum", SEARCH_SPACE["gradient_accumulation_steps"])
        lr = trial.suggest_float("learning_rate", *SEARCH_SPACE["learning_rate"], log=True)

        num_epochs = trial.suggest_categorical("num_epochs", SEARCH_SPACE["num_train_epochs"])
        if args.proxy:
            num_epochs = PROXY_EPOCHS
        elif args.quick:
            num_epochs = QUICK_EPOCHS

        warmup = trial.suggest_categorical("warmup_steps", SEARCH_SPACE["warmup_steps"])
        wd = trial.suggest_float("weight_decay", *SEARCH_SPACE["weight_decay"])
        scheduler = trial.suggest_categorical("lr_scheduler", SEARCH_SPACE["lr_scheduler_type"])
        max_seq = trial.suggest_categorical("max_seq_length", SEARCH_SPACE["max_seq_length"])

        lora_r = trial.suggest_categorical("lora_r", SEARCH_SPACE["lora_r"])
        alpha_ratio = trial.suggest_categorical("lora_alpha_ratio", SEARCH_SPACE["lora_alpha_ratio"])
        lora_alpha = int(lora_r * alpha_ratio)

        ft_vision = trial.suggest_categorical("finetune_vision", SEARCH_SPACE["finetune_vision_layers"])
        use_rslora = trial.suggest_categorical("use_rslora", SEARCH_SPACE["use_rslora"])

        tight_prob = trial.suggest_float("crop_tight_prob", *SEARCH_SPACE["crop_tight_prob"])

        effective_batch = batch_size * grad_accum
        all_params = trial.params.copy()
        all_params["lora_alpha"] = lora_alpha
        all_params["effective_batch_size"] = effective_batch

        logger.info(
            f"\n{'='*60}\n"
            f"트라이얼 {trial.number} — 파라미터:\n"
            f"  batch={batch_size}, grad_accum={grad_accum} (유효={effective_batch})\n"
            f"  lr={lr:.2e}, epochs={num_epochs}, warmup={warmup}\n"
            f"  wd={wd:.4f}, scheduler={scheduler}, rslora={use_rslora}\n"
            f"  lora_r={lora_r}, lora_alpha={lora_alpha}, vision={ft_vision}\n"
            f"  crop_tight_prob={tight_prob:.2f}\n"
            f"  max_seq_length={max_seq}\n"
            f"{'='*60}"
        )

        # ─── W&B init ─────────────────────────────────────────────────
        wandb_run = wandb_init_trial(trial.number, all_params)

        # ─── Load data (per-trial seed for independent subsets) ───────

        if args.proxy:
            data_fraction = PROXY_DATA_FRACTION
        elif args.quick:
            data_fraction = QUICK_DATA_FRACTION
        else:
            data_fraction = 1.0

        # Per-trial seed: each trial sees different data subset + crops
        random.seed(RANDOM_SEED + trial.number)

        train_dataset = load_dataset_from_jsonl(
            "train", tight_prob=tight_prob, fraction=data_fraction,
        )
        val_dataset = load_dataset_from_jsonl(
            "val", tight_prob=0.5, fraction=1.0,
        )
        logger.info(f"데이터 로딩 완료 — train: {len(train_dataset)}, val: {len(val_dataset)}")

        # ─── Load model + LoRA ────────────────────────────────────────

        clear_gpu_memory()

        model, tokenizer = FastVisionModel.from_pretrained(
            BASE_MODEL, load_in_4bit=False,
            use_gradient_checkpointing="unsloth",
        )

        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=ft_vision,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=lora_r, lora_alpha=lora_alpha,
            lora_dropout=0, bias="none",
            random_state=3407,
            use_rslora=use_rslora, loftq_config=None,
        )

        FastVisionModel.for_training(model)
        logger.info("모델 로딩 및 LoRA 적용 완료")

        # ─── Pruning callback ─────────────────────────────────────────

        class OptunaCallback(TrainerCallback):
            def __init__(self, _trial):
                self._trial = _trial
            def on_evaluate(self, _a, state, _c, metrics=None, **kw):
                if metrics and "eval_loss" in metrics:
                    step = state.global_step
                    self._trial.report(metrics["eval_loss"], step)
                    if self._trial.should_prune():
                        logger.info(f"트라이얼 {self._trial.number} 스텝 {step}에서 가지치기")
                        raise optuna.TrialPruned()

        # ─── Train ────────────────────────────────────────────────────

        report_to = "wandb" if wandb_is_available() else "none"

        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[OptunaCallback(trial)],
            args=SFTConfig(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                warmup_steps=warmup,
                num_train_epochs=num_epochs,
                learning_rate=lr, bf16=True,
                logging_steps=20,
                save_strategy="no",
                eval_strategy="epoch",
                optim="adamw_8bit",
                weight_decay=wd,
                lr_scheduler_type=scheduler,
                seed=RANDOM_SEED,
                output_dir=trial_dir,
                report_to=report_to,
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_seq_length=max_seq,
            ),
        )

        train_result = trainer.train()
        train_loss = train_result.training_loss

        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics.get("eval_loss", float("inf"))

        logger.info(f"트라이얼 {trial.number} — train_loss: {train_loss:.4f}, eval_loss: {eval_loss:.4f}")

        # ─── Optional accuracy evaluation ─────────────────────────────

        accuracy = None
        if OBJECTIVE_METRIC in ("accuracy", "combined"):
            FastVisionModel.for_inference(model)
            acc_result = evaluate_accuracy(
                model, tokenizer, val_dataset,
                max_samples=200 if (args.quick or args.proxy) else 500,
            )
            accuracy = acc_result["accuracy"]
            logger.info(f"트라이얼 {trial.number} — 정확도: {accuracy:.4f}")
            trial.set_user_attr("accuracy", accuracy)
            trial.set_user_attr("per_class_accuracy",
                                acc_result["per_class_accuracy"])

        # ─── Store metadata ───────────────────────────────────────────

        duration_min = (time.time() - trial_start) / 60
        trial.set_user_attr("train_loss", train_loss)
        trial.set_user_attr("eval_loss", eval_loss)
        trial.set_user_attr("effective_batch_size", effective_batch)
        trial.set_user_attr("lora_alpha", lora_alpha)
        trial.set_user_attr("duration_min", duration_min)

        # ─── W&B log final metrics ────────────────────────────────────

        if wandb_run:
            import wandb
            metrics = {"final/train_loss": train_loss,
                       "final/eval_loss": eval_loss,
                       "final/duration_min": duration_min}
            if accuracy is not None:
                metrics["final/accuracy"] = accuracy
            wandb.log(metrics)

        # ─── Discord notification ─────────────────────────────────────

        try:
            best_value = trial.study.best_value
            is_best = (eval_loss <= best_value)
        except ValueError:
            is_best = True  # First completed trial

        completed_count = sum(
            1 for t in trial.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        )
        discord_trial_complete(
            trial.number, eval_loss, trial.params, duration_min,
            is_best, completed_count, args.n_trials,
        )

        # ─── Return objective ─────────────────────────────────────────

        if OBJECTIVE_METRIC == "eval_loss":
            return eval_loss
        elif OBJECTIVE_METRIC == "accuracy":
            return -(accuracy or 0.0)
        elif OBJECTIVE_METRIC == "combined":
            return -(accuracy or 0.0) + 0.1 * eval_loss
        return eval_loss

    except torch.cuda.OutOfMemoryError:
        logger.warning(f"트라이얼 {trial.number} OOM 발생")
        discord_trial_pruned(trial.number,
            f"GPU 메모리 부족 (batch={trial.params.get('batch_size','?')}, "
            f"r={trial.params.get('lora_r','?')})")
        raise optuna.TrialPruned()

    except optuna.TrialPruned:
        discord_trial_pruned(trial.number, "중간 성능 부족으로 가지치기")
        raise

    except Exception as e:
        logger.error(f"트라이얼 {trial.number} 실패: {e}", exc_info=True)
        discord_trial_error(trial.number, traceback.format_exc())
        raise optuna.TrialPruned()

    finally:
        # Guaranteed cleanup on ALL code paths
        if trainer is not None:
            trainer.data_collator = None  # Sever model reference in collator
        # All vars pre-initialized to None, so these are always safe
        del trainer, model, tokenizer, train_dataset, val_dataset
        wandb_finish(wandb_run, exit_code=0)
        gc.collect()
        clear_gpu_memory()
        shutil.rmtree(trial_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════
# 8. RESULTS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def analyze_study(study: optuna.Study):
    print("\n" + "=" * 70)
    print("  하이퍼파라미터 검색 결과")
    print("=" * 70)

    best = study.best_trial
    print(f"\n  최적 트라이얼: #{best.number}")
    print(f"  최적 값: {best.value:.6f}")
    print(f"  소요 시간: {best.user_attrs.get('duration_min', 0):.1f}분")
    print(f"\n  최적 파라미터:")
    for k, v in sorted(best.params.items()):
        print(f"    {k:30s} = {v}")

    lora_r = best.params["lora_r"]
    alpha_ratio = best.params["lora_alpha_ratio"]
    print(f"    {'lora_alpha (계산됨)':30s} = {int(lora_r * alpha_ratio)}")
    bs = best.params["batch_size"]
    ga = best.params["grad_accum"]
    print(f"    {'effective_batch_size':30s} = {bs * ga}")

    if "accuracy" in best.user_attrs:
        print(f"\n  최적 정확도: {best.user_attrs['accuracy']:.4f}")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print(f"\n  트라이얼 요약:")
    print(f"    완료: {len(completed)}")
    print(f"    가지치기: {len(pruned)}")
    print(f"    실패: {len(failed)}")

    if completed:
        values = [t.value for t in completed]
        print(f"\n  목적 함수 분포:")
        print(f"    최고: {min(values):.6f}")
        print(f"    최악: {max(values):.6f}")
        print(f"    중앙값: {sorted(values)[len(values)//2]:.6f}")

    print(f"\n  상위 5개 트라이얼:")
    print(f"  {'#':>4} {'값':>10} {'학습률':>10} {'R':>4} {'배치':>4} {'에폭':>6} {'비전':>5}")
    print(f"  {'-'*50}")
    for t in sorted(completed, key=lambda t: t.value)[:5]:
        p = t.params
        print(
            f"  {t.number:4d} {t.value:10.6f} "
            f"{p['learning_rate']:10.2e} {p['lora_r']:4d} "
            f"{p['batch_size']*p['grad_accum']:4d} "
            f"{p['num_epochs']:6d} {'O' if p['finetune_vision'] else 'X':>5}"
        )

    if len(completed) >= 5:
        try:
            importance = optuna.importance.get_param_importances(study)
            print(f"\n  파라미터 중요도:")
            for param, imp in sorted(importance.items(), key=lambda x: -x[1]):
                bar = "#" * int(imp * 40)
                print(f"    {param:30s} {imp:.4f} {bar}")
        except Exception:
            print("\n  (파라미터 중요도 분석에 더 많은 트라이얼 필요)")

    # Save JSON
    results = {
        "best_trial_number": best.number, "best_value": best.value,
        "best_params": best.params, "best_user_attrs": best.user_attrs,
        "total_trials": len(study.trials),
        "completed": len(completed), "pruned": len(pruned),
        "all_trials": [
            {"number": t.number, "value": t.value,
             "params": t.params, "state": str(t.state),
             "user_attrs": t.user_attrs}
            for t in completed
        ],
    }
    results_path = os.path.join(OUTPUT_BASE, "hp_search_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  결과 저장됨: {results_path}")

    lora_alpha_val = int(best.params["lora_r"] * best.params["lora_alpha_ratio"])
    print(f"""
  최적 파라미터 요약:
  ─────────────────────────────────────────────
  per_device_train_batch_size = {best.params['batch_size']}
  gradient_accumulation_steps = {best.params['grad_accum']}
  learning_rate = {best.params['learning_rate']:.2e}
  num_train_epochs = {best.params['num_epochs']}
  warmup_steps = {best.params['warmup_steps']}
  weight_decay = {best.params['weight_decay']:.4f}
  lr_scheduler_type = "{best.params['lr_scheduler']}"
  max_seq_length = {best.params['max_seq_length']}
  lora_r = {best.params['lora_r']}
  lora_alpha = {lora_alpha_val}
  use_rslora = {best.params.get('use_rslora', False)}
  finetune_vision_layers = {best.params['finetune_vision']}
  crop_tight_prob = {best.params['crop_tight_prob']:.2f}
    """)
    print("=" * 70)
    return results


# ═══════════════════════════════════════════════════════════════════════
# 9. RETRAIN BEST MODEL
# ═══════════════════════════════════════════════════════════════════════

def retrain_best(study: optuna.Study):
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    best = study.best_trial
    p = best.params

    logger.info("최적 파라미터로 재학습 시작...")
    discord_send("🔄 **최적 파라미터로 전체 데이터 재학습을 시작합니다...**")

    model, tokenizer, trainer = None, None, None
    wandb_run = None

    try:
        random.seed(RANDOM_SEED)
        train_dataset = load_dataset_from_jsonl(
            "train", tight_prob=p["crop_tight_prob"],
        )
        val_dataset = load_dataset_from_jsonl("val")

        model, tokenizer = FastVisionModel.from_pretrained(
            BASE_MODEL, load_in_4bit=False,
            use_gradient_checkpointing="unsloth",
        )

        lora_alpha = int(p["lora_r"] * p["lora_alpha_ratio"])
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=p["finetune_vision"],
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=p["lora_r"], lora_alpha=lora_alpha,
            lora_dropout=0, bias="none",
            random_state=3407,
            use_rslora=p.get("use_rslora", False),
        )
        FastVisionModel.for_training(model)

        if wandb_is_available():
            import wandb
            wandb_run = wandb.init(
                project=WANDB_PROJECT, entity=WANDB_ENTITY or None,
                name="retrain-best", group=STUDY_NAME,
                config=p, reinit=True, tags=["retrain", "best-model"],
            )

        report_to = "wandb" if wandb_is_available() else "none"

        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=train_dataset, eval_dataset=val_dataset,
            args=SFTConfig(
                per_device_train_batch_size=p["batch_size"],
                gradient_accumulation_steps=p["grad_accum"],
                warmup_steps=p["warmup_steps"],
                num_train_epochs=p["num_epochs"],
                learning_rate=p["learning_rate"],
                bf16=True, logging_steps=10,
                save_strategy="epoch", eval_strategy="epoch",
                optim="adamw_8bit",
                weight_decay=p["weight_decay"],
                lr_scheduler_type=p["lr_scheduler"],
                seed=RANDOM_SEED,
                output_dir=BEST_MODEL_DIR,
                report_to=report_to,
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_seq_length=p["max_seq_length"],
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
            ),
        )

        trainer.train()

        lora_path = os.path.join(BEST_MODEL_DIR, "lora")
        model.save_pretrained(lora_path)
        tokenizer.save_pretrained(lora_path)

        FastVisionModel.for_inference(model)
        acc_result = evaluate_accuracy(model, tokenizer, val_dataset,
                                       max_samples=500)
        logger.info(f"최종 정확도: {acc_result['accuracy']:.4f}")

        if wandb_run:
            import wandb
            wandb.log({"retrain/accuracy": acc_result["accuracy"]})

        discord_retrain_complete(acc_result["accuracy"], lora_path)
        print(f"\n  최적 모델 저장됨: {lora_path}")
        print(f"  최종 정확도: {acc_result['accuracy']:.2%}")
        return acc_result

    finally:
        if trainer is not None:
            trainer.data_collator = None
        for obj_name in ("trainer", "model", "tokenizer"):
            obj = locals().get(obj_name)
            if obj is not None:
                del obj
        wandb_finish(wandb_run)
        gc.collect()
        clear_gpu_memory()


# ═══════════════════════════════════════════════════════════════════════
# 10. MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="해충 탐지 LoRA 파인튜닝 하이퍼파라미터 검색"
    )
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    parser.add_argument("--quick", action="store_true",
                        help="빠른 모드: 1 에폭, 20%% 데이터")
    parser.add_argument("--proxy", action="store_true",
                        help="프록시 모드: 1 에폭, 10%% 데이터")
    parser.add_argument("--analyze", action="store_true",
                        help="기존 결과 분석만 수행")
    parser.add_argument("--retrain", action="store_true",
                        help="최적 파라미터로 재학습")
    parser.add_argument("--metric", choices=["eval_loss", "accuracy", "combined"],
                        default=None)
    args = parser.parse_args()

    if args.proxy:
        args.quick = False

    global OBJECTIVE_METRIC
    if args.metric:
        OBJECTIVE_METRIC = args.metric

    # ─── Setup logging (deferred to avoid import-time crash) ──────────
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )

    # ─── Analyze only ─────────────────────────────────────────────────

    if args.analyze:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_DB)
        analyze_study(study)
        wandb_log_best_summary(study)
        return

    # ─── Create / load study ──────────────────────────────────────────

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_DB,
        direction="minimize",
        sampler=TPESampler(
            seed=RANDOM_SEED, n_startup_trials=5, multivariate=True,
        ),
        pruner=HyperbandPruner(
            min_resource=1, max_resource=5, reduction_factor=3,
        ),
        load_if_exists=True,
    )

    mode = ("PROXY (10%)" if args.proxy
            else "QUICK (20%)" if args.quick
            else "FULL")

    logger.info(
        f"\n{'='*60}\n"
        f"  하이퍼파라미터 검색\n"
        f"  Study: {STUDY_NAME}\n"
        f"  목적 함수: {OBJECTIVE_METRIC}\n"
        f"  트라이얼: {args.n_trials}\n"
        f"  모드: {mode}\n"
        f"  W&B: {'활성화' if wandb_is_available() else '비활성화'}\n"
        f"  Discord: {'활성화' if DISCORD_WEBHOOK_URL else '비활성화'}\n"
        f"{'='*60}"
    )

    discord_search_started(args.n_trials, mode)

    # ─── Run optimization ─────────────────────────────────────────────

    search_start = time.time()

    try:
        study.optimize(
            lambda trial: objective(trial, args),
            n_trials=args.n_trials,
        )
    except Exception as e:
        discord_error(traceback.format_exc())
        raise

    search_hours = (time.time() - search_start) / 3600

    # ─── Analyze ──────────────────────────────────────────────────────

    results = analyze_study(study)
    wandb_log_best_summary(study)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    discord_phase_complete(
        phase=f"{mode} 검색",
        best_value=study.best_trial.value,
        best_params=study.best_trial.params,
        completed=len(completed), pruned=len(pruned),
        total_time_hr=search_hours,
    )

    # ─── Retrain ──────────────────────────────────────────────────────

    if args.retrain:
        try:
            retrain_best(study)
        except Exception as e:
            discord_error(f"재학습 실패: {traceback.format_exc()}")
            raise

    # ─── Visualizations ───────────────────────────────────────────────

    try:
        from optuna.visualization import (
            plot_optimization_history, plot_param_importances,
            plot_parallel_coordinate, plot_slice,
        )
        for name, fn in [
            ("optimization_history", plot_optimization_history),
            ("param_importance", plot_param_importances),
            ("parallel_coordinate", plot_parallel_coordinate),
            ("param_slices", plot_slice),
        ]:
            fig = fn(study)
            fig.write_html(os.path.join(OUTPUT_BASE, f"{name}.html"))
        logger.info(f"시각화 저장됨: {OUTPUT_BASE}/")
    except ImportError:
        logger.info("pip install plotly kaleido 로 시각화 설치")

    logger.info("검색 완료!")


if __name__ == "__main__":
    main()
