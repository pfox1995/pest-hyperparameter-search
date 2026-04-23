#!/usr/bin/env python3
"""Baseline eval — base unsloth/Qwen3.5-9B model WITHOUT the LoRA adapter.

Identical pipeline to eval_v8.py (letterbox 512x512, fixed prompt, fixed 16
output tokens, PREFIX + substring normalization, NanumGothic CM plot) except
PEFT/adapter is never attached. Produces a direct apples-to-apples baseline
for comparing against the fine-tuned pfox1995/pest-detector-final adapter.
"""
import argparse
import json
import logging
import os
import sys
import time

os.environ.setdefault("HP_LAZY_DATASET", "1")
sys.path.insert(0, "/workspace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

import torch
import torch._dynamo

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 16384
torch._dynamo.config.accumulated_recompile_limit = 16384
for attr in ("error_on_recompile", "fail_on_recompile_limit_hit"):
    if hasattr(torch._dynamo.config, attr):
        setattr(torch._dynamo.config, attr, False)

from PIL import Image  # noqa: E402
from hp_search import BASE_MODEL, SYSTEM_MSG  # noqa: E402


FIXED_PROMPT = "이 사진에 있는 해충의 이름을 알려주세요."
FIXED_SIZE = 512
FILL_COLOR = (128, 128, 128)
MAX_NEW_TOKENS = 12  # pest names are 2-9 tokens; 12 is plenty


def letterbox(img: Image.Image, size: int = FIXED_SIZE) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    r = img.resize((nw, nh), Image.Resampling.LANCZOS)
    c = Image.new("RGB", (size, size), FILL_COLOR)
    c.paste(r, ((size - nw) // 2, (size - nh) // 2))
    return c


def iter_val_samples():
    data_dir = os.environ.get("HP_DATA_DIR", "/workspace/data")
    jsonl_path = os.path.join(data_dir, "val.jsonl")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            msgs = rec["messages"]
            truth = msgs[-1]["content"][0]["text"]
            img_rel = None
            for m in msgs:
                for c in m["content"]:
                    if c.get("type") == "image" and "image" in c:
                        img_rel = c["image"]
                        break
                if img_rel:
                    break
            if not img_rel:
                continue
            img_rel = img_rel.replace("\\", "/")
            img_path = os.path.join(data_dir, img_rel)
            if not os.path.exists(img_path):
                continue
            try:
                with Image.open(img_path) as raw:
                    img = letterbox(raw, FIXED_SIZE)
            except Exception as e:
                log.warning(f"skip {img_path}: {e}")
                continue
            yield truth, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--save-dir", default="/workspace/hf_eval_baseline")
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--max-samples", type=int, default=-1)
    args = ap.parse_args()

    from unsloth import FastVisionModel

    log.info(f"Loading base model (NO adapter): {args.base}")
    model, tok = FastVisionModel.from_pretrained(args.base, load_in_4bit=False)

    torch._dynamo.config.suppress_errors = True
    for attr in ("error_on_recompile", "fail_on_recompile_limit_hit"):
        if hasattr(torch._dynamo.config, attr):
            setattr(torch._dynamo.config, attr, False)

    model.eval()
    model.forward = torch._dynamo.disable(model.forward)
    log.info("model.forward wrapped with torch._dynamo.disable → pure eager")
    log.info("NOTE: No LoRA adapter attached — evaluating raw base model")

    log.info("Materializing val samples (letterbox 512x512, fixed prompt)...")
    t0 = time.time()
    all_samples = list(iter_val_samples())
    if args.max_samples > 0:
        all_samples = all_samples[: args.max_samples]
    n = len(all_samples)
    log.info(f"Val samples prepared: {n} (took {time.time()-t0:.1f}s)")

    pad_id = (
        tok.tokenizer.pad_token_id
        if hasattr(tok, "tokenizer") and tok.tokenizer.pad_token_id is not None
        else (tok.tokenizer.eos_token_id if hasattr(tok, "tokenizer") else None)
    )

    y_true, y_pred_raw = [], []
    t_start = time.time()

    for idx, (truth, image) in enumerate(all_samples):
        infer = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": FIXED_PROMPT},
            ]},
        ]
        gen = ""
        for attempt in (1, 2):
            try:
                tmpl = tok.apply_chat_template(infer, add_generation_prompt=True)
                inp = tok(
                    image, tmpl,
                    add_special_tokens=False, return_tensors="pt",
                ).to("cuda")
                with torch.inference_mode():
                    kwargs = dict(
                        max_new_tokens=MAX_NEW_TOKENS,
                        use_cache=True,
                        do_sample=False,
                    )
                    if pad_id is not None:
                        kwargs["pad_token_id"] = pad_id
                    out = model.generate(**inp, **kwargs)
                gen = tok.decode(
                    out[0][inp["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()
                del inp, out
                break
            except Exception as e:
                log.warning(
                    f"idx={idx} attempt {attempt}: "
                    f"{type(e).__name__}: {str(e)[:160]}"
                )
                try:
                    torch._dynamo.reset()
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        y_true.append(truth)
        y_pred_raw.append(gen)

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            eta_min = (n - (idx + 1)) / rate / 60
            log.info(
                f"progress {idx+1}/{n} "
                f"({rate:.2f}/s, ETA {eta_min:.1f}m) "
                f"last={gen[:40]!r}"
            )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # PREFIX + substring normalize
    REAL_CLASSES = sorted(set(y_true))
    classes_desc = sorted(REAL_CLASSES, key=len, reverse=True)

    def normalize_with_unknown(g: str):
        g = g.strip()
        for c in classes_desc:
            if g.startswith(c):
                return c, True
        for c in classes_desc:
            if c in g:
                return c, True
        return "UNKNOWN", False

    results = [normalize_with_unknown(g) for g in y_pred_raw]
    y_pred_full = [r[0] for r in results]
    in_vocab = [r[1] for r in results]
    n_unknown = sum(1 for x in in_vocab if not x)

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix,
    )

    # Views 1: with UNKNOWN bucket (all 1595 samples)
    labels_with_unknown = REAL_CLASSES + ["UNKNOWN"]
    acc_all = accuracy_score(y_true, y_pred_full)
    f1m_all = f1_score(y_true, y_pred_full, labels=REAL_CLASSES,
                       average="macro", zero_division=0)
    f1w_all = f1_score(y_true, y_pred_full, labels=REAL_CLASSES,
                       average="weighted", zero_division=0)
    pm_all = precision_score(y_true, y_pred_full, labels=REAL_CLASSES,
                             average="macro", zero_division=0)
    rm_all = recall_score(y_true, y_pred_full, labels=REAL_CLASSES,
                          average="macro", zero_division=0)
    cm_all = confusion_matrix(y_true, y_pred_full, labels=labels_with_unknown)

    # View 2: off-vocab dropped
    y_true_f = [t for t, ok in zip(y_true, in_vocab) if ok]
    y_pred_f = [p for p, ok in zip(y_pred_full, in_vocab) if ok]
    total_f = len(y_true_f)
    acc_f = accuracy_score(y_true_f, y_pred_f) if total_f else 0.0
    f1m_f = f1_score(y_true_f, y_pred_f, labels=REAL_CLASSES,
                     average="macro", zero_division=0) if total_f else 0.0
    f1w_f = f1_score(y_true_f, y_pred_f, labels=REAL_CLASSES,
                     average="weighted", zero_division=0) if total_f else 0.0
    pm_f = precision_score(y_true_f, y_pred_f, labels=REAL_CLASSES,
                           average="macro", zero_division=0) if total_f else 0.0
    rm_f = recall_score(y_true_f, y_pred_f, labels=REAL_CLASSES,
                        average="macro", zero_division=0) if total_f else 0.0
    cm_f = (confusion_matrix(y_true_f, y_pred_f, labels=REAL_CLASSES).tolist()
            if total_f else [])

    p_per = precision_score(y_true, y_pred_full, labels=REAL_CLASSES,
                            average=None, zero_division=0)
    r_per = recall_score(y_true, y_pred_full, labels=REAL_CLASSES,
                         average=None, zero_division=0)
    f_per = f1_score(y_true, y_pred_full, labels=REAL_CLASSES,
                     average=None, zero_division=0)
    per_class = {
        cls: {
            "precision": float(p_per[i]),
            "recall": float(r_per[i]),
            "f1": float(f_per[i]),
            "support": int(y_true.count(cls)),
        }
        for i, cls in enumerate(REAL_CLASSES)
    }

    os.makedirs(args.save_dir, exist_ok=True)

    # Plot CM (real classes + UNKNOWN column)
    cm_path = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        kws = ["nanum", "malgun", "gothic", "gulim", "noto", "cjk"]
        def _fonts():
            return [f for f in font_manager.findSystemFonts()
                    if any(k in f.lower() for k in kws)]
        fonts = _fonts()
        if not fonts:
            font_manager._load_fontmanager(try_read_cache=False)
            fonts = _fonts()
        if fonts:
            plt.rcParams["font.family"] = font_manager.FontProperties(
                fname=fonts[0]).get_name()
        plt.rcParams["axes.unicode_minus"] = False

        n_lbl = len(labels_with_unknown)
        fs = max(11, n_lbl * 0.55)
        fig, ax = plt.subplots(figsize=(fs, fs))
        im = ax.imshow(cm_all, cmap="Blues", interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(n_lbl))
        ax.set_yticks(range(n_lbl))
        ax.set_xticklabels(labels_with_unknown, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels_with_unknown, fontsize=9)
        ax.set_xlabel("예측 (Predicted)")
        ax.set_ylabel("실제 (Actual)")
        ax.set_title(
            f"Base Qwen3.5-9B (no LoRA) — Confusion Matrix\n"
            f"Accuracy={acc_all:.4f}   F1_macro={f1m_all:.4f}   "
            f"F1_weighted={f1w_all:.4f}   off_vocab={n_unknown}"
        )
        thr = cm_all.max() / 2 if cm_all.max() > 0 else 1
        for i in range(n_lbl):
            for j in range(n_lbl):
                if cm_all[i, j] > 0:
                    ax.text(
                        j, i, str(cm_all[i, j]),
                        ha="center", va="center", fontsize=8,
                        color="white" if cm_all[i, j] > thr else "black",
                    )
        plt.tight_layout()
        cm_path = os.path.join(args.save_dir, "confusion_matrix_baseline.png")
        fig.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.warning(f"CM plot failed: {e}")

    result = {
        "model": f"{args.base} (base, no LoRA adapter)",
        "base_model": args.base,
        "total": len(y_true),
        "off_vocab_count": n_unknown,
        # View 1: include UNKNOWN in samples, exclude from class averaging
        "accuracy": acc_all,
        "precision_macro": pm_all, "recall_macro": rm_all, "f1_macro": f1m_all,
        "f1_weighted": f1w_all,
        # View 2: off-vocab dropped
        "filtered": {
            "total": total_f,
            "accuracy": acc_f,
            "precision_macro": pm_f, "recall_macro": rm_f,
            "f1_macro": f1m_f, "f1_weighted": f1w_f,
            "confusion_matrix": cm_f,
        },
        "per_class": per_class,
        "labels": REAL_CLASSES,
        "labels_with_unknown": labels_with_unknown,
        "confusion_matrix": cm_all.tolist(),
        "confusion_matrix_path": cm_path,
    }

    with open(os.path.join(args.save_dir, "metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str, ensure_ascii=False)
    with open(os.path.join(args.save_dir, "raw_predictions.json"), "w",
              encoding="utf-8") as f:
        json.dump(
            [{"truth": t, "raw": r, "normalized": n, "in_vocab": ok}
             for t, r, n, ok in zip(y_true, y_pred_raw, y_pred_full, in_vocab)],
            f, indent=2, ensure_ascii=False,
        )

    total_time = time.time() - t_start
    print("=" * 70)
    print(f"BASELINE (no LoRA adapter)")
    print(f"Base:       {args.base}")
    print(f"Samples:    {len(y_true)}    (elapsed {total_time/60:.1f} min)")
    print(f"Off-vocab:  {n_unknown} ({n_unknown/len(y_true)*100:.1f}%)")
    print(f"")
    print(f"All samples (UNKNOWN counted as wrong):")
    print(f"  Accuracy:    {acc_all:.4f}")
    print(f"  F1 macro:    {f1m_all:.4f}   F1 weighted: {f1w_all:.4f}")
    print(f"  Prec macro:  {pm_all:.4f}   Rec macro:   {rm_all:.4f}")
    print(f"")
    print(f"Off-vocab dropped ({total_f} samples):")
    print(f"  Accuracy:    {acc_f:.4f}")
    print(f"  F1 macro:    {f1m_f:.4f}   F1 weighted: {f1w_f:.4f}")
    print(f"  Prec macro:  {pm_f:.4f}   Rec macro:   {rm_f:.4f}")
    print(f"")
    print(f"CM PNG:     {cm_path}")
    print(f"Metrics:    {os.path.join(args.save_dir, 'metrics.json')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
