#!/usr/bin/env python3
"""
Confusion-matrix eval for a HuggingFace-hosted LoRA adapter.
============================================================
Loads the base Qwen3.5-9B vision model + a LoRA adapter from HF Hub,
runs generation-based eval on val.jsonl, writes:
  - confusion_matrix_trial_<label>.png   (Korean-safe font)
  - metrics.json                         (per-class P/R/F1, CM, acc)

Defaults target the user's published adapter + dataset layout used by
the training pipeline. The adapter argument accepts EITHER an HF repo
ID (e.g. "pfox1995/pest-detector-final") OR a local directory.

Usage (from repo root on RunPod):
    # Full val set, default adapter
    python3 cm_from_hf.py

    # Quick smoke test on 100 samples
    python3 cm_from_hf.py --max-samples 100

    # Custom adapter (HF repo or local path)
    python3 cm_from_hf.py --adapter pfox1995/pest-detector-final

    # Different base model
    python3 cm_from_hf.py --base-model unsloth/Qwen3.5-9B

Prerequisites:
    - Dataset at $HP_DATA_DIR (default /workspace/data), populated by
      download_dataset.py (val.jsonl + image files).
    - HF_TOKEN only needed if the adapter repo is private.
"""
import argparse
import json
import logging
import os
import sys

# Make hp_search importable regardless of cwd
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

# Lazy dataset = decode-on-access; matches eval_only.py / train_final.py usage
os.environ.setdefault("HP_LAZY_DATASET", "1")

# Basic logging so hp_search.logger messages surface when imported as a library
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from hp_search import (  # noqa: E402
    BASE_MODEL,
    SYSTEM_MSG,
    load_dataset_from_jsonl,
    logger,
)

DEFAULT_ADAPTER = "pfox1995/pest-detector-final"


# ══════════════════════════════════════════════════════════════════════
# USER HOOK — label matching policy (fill this in for your use case)
# ══════════════════════════════════════════════════════════════════════
def normalize_prediction(generated: str, class_labels: list) -> str:
    """Map a raw model generation to a label string counted in the CM.

    The model is trained to emit exactly one of the class labels in
    `class_labels` (Korean pest names like "배추흰나비", plus "정상" for
    no-pest images). Greedy decoding with max_new_tokens=10 means the
    output can be:

      1. An exact match                       → "배추흰나비"
      2. The right label + trailing tokens    → "배추흰나비입니다"
      3. A truncated label (rare, long names) → "배추흰나"
      4. A wrong-but-plausible class          → "파밤나방"
      5. Off-vocab gibberish                  → "곤충의" / empty / emoji

    Policies you can implement (pick one or mix):
      * STRICT  — return generated.strip()
                  predictable, matches hp_search.evaluate_model; under-
                  counts cases 2 and 3 as wrong even when the intent is
                  clearly right.
      * PREFIX  — if any class_label is a prefix of generated, return
                  that class; otherwise generated.strip()
                  rescues case 2 cheaply; case 3 still fails.
      * FUZZY   — snap to the class with smallest edit distance to
                  generated (e.g. difflib.get_close_matches with a
                  cutoff). Rescues cases 2 and 3; risk is rewarding
                  confident-but-wrong predictions in case 4 by snapping
                  them to a nearby-but-incorrect class.
      * BUCKET  — anything not in class_labels → "UNKNOWN". Keeps the CM
                  honest about off-vocab failures at the cost of adding
                  a 20th row/column.

    Trade-off summary: STRICT is a lower bound on true accuracy; FUZZY
    is an upper bound but can mask real mistakes. For reporting to a
    paper/stakeholders, STRICT + a separate PREFIX score is a good pair.

    Args:
        generated:    Raw decoded text from model.generate(), already
                      .strip()'d of whitespace but NOT of trailing noise.
        class_labels: Sorted list of the ground-truth classes observed
                      in this eval run (≤19 + "정상").

    Returns:
        The label string that will be counted in the confusion matrix.

    Chosen policy: STRICT.
        Rationale: matches hp_search.evaluate_model exactly, so the CM
        produced here is directly comparable to the end-of-training eval
        numbers. Cases 2/3/5 above are counted as wrong — this is a
        conservative lower bound on true accuracy. raw_predictions.json
        is saved alongside metrics.json, so a looser policy can be
        re-scored later without re-running the ~25 min generation.
    """
    return generated.strip()


# ══════════════════════════════════════════════════════════════════════
# Generation loop — one sample at a time, same as training-time eval
# ══════════════════════════════════════════════════════════════════════
def run_generation(model, tokenizer, val_dataset, n_eval):
    """Run generation over the first n_eval val samples.

    Returns (y_true, y_pred_raw). Both lists are string labels. Raw
    means pre-normalize_prediction — the caller applies the user's
    policy before computing metrics.
    """
    import torch

    samples = val_dataset[:n_eval]
    y_true, y_pred_raw = [], []

    for idx, item in enumerate(samples):
        messages = item["messages"]
        truth = messages[-1]["content"][0]["text"]
        image = messages[1]["content"][0]["image"]
        user_text = messages[1]["content"][1]["text"]

        infer_messages = [
            {"role": "system", "content": [
                {"type": "text", "text": SYSTEM_MSG}
            ]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ]},
        ]

        try:
            input_text = tokenizer.apply_chat_template(
                infer_messages, add_generation_prompt=True,
            )
            inputs = tokenizer(
                image, input_text,
                add_special_tokens=False, return_tensors="pt",
            ).to("cuda")

            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=10, use_cache=True,
                )

            generated = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            del inputs, out

            y_true.append(truth)
            y_pred_raw.append(generated)

        except Exception as e:
            logger.warning(f"추론 오류 idx={idx}: {e}")
            continue

        if (idx + 1) % 50 == 0:
            logger.info(f"진행 {idx + 1}/{len(samples)} — 마지막 예측: {generated!r}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return y_true, y_pred_raw


# ══════════════════════════════════════════════════════════════════════
# Metrics + CM plotting (Korean-font-safe, mirrors hp_search.evaluate_model)
# ══════════════════════════════════════════════════════════════════════
def compute_and_plot(y_true, y_pred, save_dir, label):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix,
    )

    all_labels = sorted(set(y_true + y_pred))
    acc = accuracy_score(y_true, y_pred)
    prec_m = precision_score(y_true, y_pred, labels=all_labels,
                             average="macro", zero_division=0)
    rec_m = recall_score(y_true, y_pred, labels=all_labels,
                         average="macro", zero_division=0)
    f1_m = f1_score(y_true, y_pred, labels=all_labels,
                    average="macro", zero_division=0)
    f1_w = f1_score(y_true, y_pred, labels=all_labels,
                    average="weighted", zero_division=0)

    prec_per = precision_score(y_true, y_pred, labels=all_labels,
                               average=None, zero_division=0)
    rec_per = recall_score(y_true, y_pred, labels=all_labels,
                           average=None, zero_division=0)
    f1_per = f1_score(y_true, y_pred, labels=all_labels,
                      average=None, zero_division=0)
    per_class = {
        cls: {
            "precision": float(prec_per[i]),
            "recall": float(rec_per[i]),
            "f1": float(f1_per[i]),
            "support": int(y_true.count(cls)),
        }
        for i, cls in enumerate(all_labels)
    }

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    os.makedirs(save_dir, exist_ok=True)
    cm_path = None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        # Korean font resolution: matplotlib caches fonts at import, so a
        # font installed later won't be seen unless we rebuild the cache.
        _kw = ["nanum", "malgun", "gothic", "gulim", "noto", "cjk"]
        def _find():
            return [f for f in font_manager.findSystemFonts()
                    if any(k in f.lower() for k in _kw)]

        fonts = _find()
        if not fonts:
            font_manager._load_fontmanager(try_read_cache=False)
            fonts = _find()
        if fonts:
            plt.rcParams["font.family"] = font_manager.FontProperties(
                fname=fonts[0]).get_name()
        plt.rcParams["axes.unicode_minus"] = False

        short = [l[:4] for l in all_labels]
        n = len(all_labels)
        fs = max(8, n * 0.6)
        fig, ax = plt.subplots(figsize=(fs, fs))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(short, fontsize=7)
        ax.set_xlabel("예측 (Predicted)")
        ax.set_ylabel("실제 (Actual)")
        ax.set_title(
            f"Confusion Matrix ({label})\n"
            f"Acc={acc:.3f}  F1(macro)={f1_m:.3f}  n={len(y_true)}"
        )
        thr = cm.max() / 2
        for i in range(n):
            for j in range(n):
                if cm[i, j] > 0:
                    ax.text(j, i, str(cm[i, j]),
                            ha="center", va="center", fontsize=6,
                            color="white" if cm[i, j] > thr else "black")
        plt.tight_layout()
        cm_path = os.path.join(save_dir, f"confusion_matrix_trial_{label}.png")
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        logger.info(f"혼동 행렬 저장됨: {cm_path}")
    except Exception as e:
        logger.warning(f"혼동 행렬 플롯 실패: {e}")

    return {
        "accuracy": acc,
        "precision_macro": prec_m,
        "recall_macro": rec_m,
        "f1_macro": f1_m,
        "f1_weighted": f1_w,
        "per_class": per_class,
        "labels": all_labels,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_path": cm_path,
        "total": len(y_true),
        "correct": int(acc * len(y_true)),
    }


def resolve_adapter(adapter: str) -> str:
    """Return a local adapter dir. Downloads from HF Hub if needed."""
    if os.path.isdir(adapter):
        return adapter
    logger.info(f"'{adapter}' is not a local dir — resolving as HF repo ID")
    from huggingface_hub import snapshot_download
    token = os.environ.get("HF_TOKEN") or None
    path = snapshot_download(repo_id=adapter, token=token)
    logger.info(f"Adapter downloaded to: {path}")
    return path


def main():
    p = argparse.ArgumentParser(
        description="Confusion matrix from a HF-hosted LoRA adapter.",
    )
    p.add_argument(
        "--adapter", default=DEFAULT_ADAPTER,
        help=f"HF repo ID or local directory. Default: {DEFAULT_ADAPTER}",
    )
    p.add_argument(
        "--base-model", default=BASE_MODEL,
        help=f"Base model HF repo ID. Default: {BASE_MODEL}",
    )
    p.add_argument(
        "--max-samples", type=int, default=-1,
        help="Max val samples to evaluate. -1 = full val set.",
    )
    p.add_argument(
        "--save-dir", default="/workspace/hf_eval",
        help="Where to write the CM PNG + metrics.json.",
    )
    p.add_argument(
        "--label", default="hf",
        help="Tag for the CM file name: confusion_matrix_trial_<label>.png",
    )
    args = p.parse_args()

    from unsloth import FastVisionModel

    logger.info(f"Loading base model: {args.base_model}")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.base_model, load_in_4bit=False,
    )

    adapter_path = resolve_adapter(args.adapter)
    logger.info(f"Attaching adapter: {adapter_path}")
    model.load_adapter(adapter_path)
    FastVisionModel.for_inference(model)

    logger.info("Loading val dataset...")
    val_dataset = load_dataset_from_jsonl("val")
    n_val = len(val_dataset)
    n_eval = n_val if args.max_samples < 0 else min(args.max_samples, n_val)
    logger.info(f"Val samples: {n_val}, evaluating on {n_eval}")

    y_true, y_pred_raw = run_generation(model, tokenizer, val_dataset, n_eval)

    class_labels = sorted(set(y_true))
    y_pred = [normalize_prediction(g, class_labels) for g in y_pred_raw]

    result = compute_and_plot(y_true, y_pred, args.save_dir, args.label)

    # Persist raw predictions too — useful for tuning normalize_prediction
    # without re-running generation
    with open(os.path.join(args.save_dir, "raw_predictions.json"), "w",
              encoding="utf-8") as f:
        json.dump(
            [{"truth": t, "raw": r, "normalized": n}
             for t, r, n in zip(y_true, y_pred_raw, y_pred)],
            f, indent=2, ensure_ascii=False,
        )

    with open(os.path.join(args.save_dir, "metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str, ensure_ascii=False)

    print("=" * 60)
    print(f"Adapter:     {args.adapter}")
    print(f"Base model:  {args.base_model}")
    print(f"Samples:     {result['total']}")
    print(f"Accuracy:    {result['accuracy']:.4f}")
    print(f"F1 macro:    {result['f1_macro']:.4f}")
    print(f"F1 weighted: {result['f1_weighted']:.4f}")
    print(f"Precision:   {result['precision_macro']:.4f}")
    print(f"Recall:      {result['recall_macro']:.4f}")
    print(f"CM PNG:      {result.get('confusion_matrix_path', 'n/a')}")
    print(f"Metrics:     {os.path.join(args.save_dir, 'metrics.json')}")
    print(f"Raw preds:   {os.path.join(args.save_dir, 'raw_predictions.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
