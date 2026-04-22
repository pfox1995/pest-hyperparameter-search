#!/usr/bin/env python3
"""
Generate a confusion matrix + classification metrics for a saved LoRA
adapter, without re-training.
=====================================================================

Use cases:
  * Training was killed mid-run — produce the end-of-run eval artifacts
    that the aborted trainer never got to run.
  * You want a confusion matrix over the FULL val set (1595 samples)
    instead of the --eval-samples cap that train_final.py applies.
  * You want to re-score a historical checkpoint without retraining.

Assumes the repo's hp_search.py is importable (this script lives in the
same directory). Reuses evaluate_model() from there so the CM, metrics,
and per-class output are byte-identical to the end-of-training eval.

Usage (after killing the training):

    # Evaluate the golden backup (recommended)
    python3 eval_only.py

    # Evaluate a specific checkpoint
    python3 eval_only.py --adapter /workspace/best-pest-detector/checkpoint-525

    # Cap the sample count (useful for a quick smoke test)
    python3 eval_only.py --max-samples 200

    # Also push results to HF Hub
    python3 eval_only.py --push-to-hub

Output:
    --save-dir/confusion_matrix_trial_eval-only.png
    --save-dir/metrics.json
"""
import argparse
import json
import os
import sys

# ─── Make hp_search importable regardless of CWD ──────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

# Default to lazy dataset — evaluate_model needs a val_dataset that
# supports slicing; LazyImageDataset (+ our slice-aware __getitem__
# fix) is the one both train_final.py and train.py use in practice.
os.environ.setdefault("HP_LAZY_DATASET", "1")

from hp_search import (  # noqa: E402
    BASE_MODEL,
    evaluate_model,
    load_dataset_from_jsonl,
    logger,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run confusion matrix eval on a saved LoRA adapter.",
    )
    parser.add_argument(
        "--adapter", default="/workspace/_golden/best_ckpt",
        help="Path to the LoRA adapter directory (contains "
             "adapter_model.safetensors + adapter_config.json). "
             "Default: /workspace/_golden/best_ckpt.",
    )
    parser.add_argument(
        "--base-model", default=BASE_MODEL,
        help=f"Base model HF repo ID. Default: {BASE_MODEL}.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=-1,
        help="Max val samples to evaluate. -1 = full val set "
             "(recommended for reliable per-class F1 on 19 classes).",
    )
    parser.add_argument(
        "--save-dir", default="/workspace/golden_eval",
        help="Directory to save the confusion matrix PNG + metrics.json.",
    )
    parser.add_argument(
        "--trial-num", default="eval-only",
        help="Label for the CM file name (confusion_matrix_trial_<label>.png).",
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="After local save, upload the eval folder to HF Hub at "
             "$HF_REPO_ID/evaluation/.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.adapter):
        logger.error(f"Adapter directory not found: {args.adapter}")
        sys.exit(1)

    os.makedirs(args.save_dir, exist_ok=True)

    # ─── Load base model + adapter ────────────────────────────────────
    logger.info(f"Loading base model: {args.base_model}")
    from unsloth import FastVisionModel
    import torch

    model, tokenizer = FastVisionModel.from_pretrained(
        args.base_model,
        load_in_4bit=False,
    )

    logger.info(f"Attaching adapter: {args.adapter}")
    model.load_adapter(args.adapter)
    FastVisionModel.for_inference(model)

    # ─── Load val set ─────────────────────────────────────────────────
    logger.info("Loading val dataset...")
    val_dataset = load_dataset_from_jsonl("val")
    n_val = len(val_dataset)
    n_eval = n_val if args.max_samples < 0 else min(args.max_samples, n_val)
    logger.info(f"Val samples: {n_val}, evaluating on {n_eval}")

    # ─── Run eval (generation-based, one sample at a time) ────────────
    result = evaluate_model(
        model, tokenizer, val_dataset,
        max_samples=n_eval,
        save_dir=args.save_dir,
        trial_num=args.trial_num,
    )

    # ─── Report ───────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Adapter:   {args.adapter}")
    print(f"Samples:   {result.get('total', n_eval)}")
    print(f"Accuracy:  {result['accuracy']:.4f}")
    print(f"F1 macro:  {result['f1_macro']:.4f}")
    print(f"F1 weighted: {result['f1_weighted']:.4f}")
    print(f"Precision: {result['precision_macro']:.4f}")
    print(f"Recall:    {result['recall_macro']:.4f}")
    print(f"CM PNG:    {result.get('confusion_matrix_path', 'n/a')}")
    print(f"Metrics:   {args.save_dir}/metrics.json")
    print("=" * 60)

    # Save metrics.json alongside the CM image
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2, default=str, ensure_ascii=False)

    # ─── Optional HF Hub push ─────────────────────────────────────────
    if args.push_to_hub:
        hf_repo = os.environ.get("HF_REPO_ID", "")
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_repo or not hf_token:
            logger.warning(
                "HF_REPO_ID or HF_TOKEN not set — skipping HF upload."
            )
        else:
            from huggingface_hub import upload_folder
            logger.info(f"Uploading {args.save_dir} → {hf_repo}/evaluation/")
            upload_folder(
                folder_path=args.save_dir,
                path_in_repo="evaluation",
                repo_id=hf_repo,
                token=hf_token,
            )
            logger.info(
                f"Uploaded: https://huggingface.co/{hf_repo}/tree/main/evaluation"
            )


if __name__ == "__main__":
    main()
