#!/usr/bin/env python3
"""V6 eval — targets all 1595 samples by disabling compile per-forward.

Strategy:
  1. Load base + adapter via Unsloth (known-good path — v2 gave 94.29%)
  2. AFTER load, wrap model.forward with torch._dynamo.disable so every
     forward call runs in eager mode, bypassing Unsloth's compiled kernels
     that cause accumulated_recompile_limit drops on variable image shapes.
  3. Retry-once with torch._dynamo.reset() on any exception.
  4. Even on final failure, append empty string so sample count matches.

Writes to --save-dir:
  confusion_matrix_v6.png (Korean-safe via NanumGothic)
  metrics.json            (acc, P/R/F1 macro+weighted, per-class, CM)
  raw_predictions.json    (truth/raw/normalized triples)
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

# Set BEFORE any Unsloth/transformers import
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 8192
torch._dynamo.config.accumulated_recompile_limit = 8192
if hasattr(torch._dynamo.config, "error_on_recompile"):
    torch._dynamo.config.error_on_recompile = False

from hp_search import BASE_MODEL, SYSTEM_MSG, load_dataset_from_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="pfox1995/pest-detector-final")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--save-dir", default="/workspace/hf_eval_v6")
    ap.add_argument("--label", default="v6")
    ap.add_argument("--max-samples", type=int, default=-1)
    args = ap.parse_args()

    from unsloth import FastVisionModel
    from peft import PeftModel
    from huggingface_hub import snapshot_download

    log.info(f"Loading base: {args.base}")
    model, tok = FastVisionModel.from_pretrained(args.base, load_in_4bit=False)

    # Re-apply dynamo config AFTER Unsloth load — Unsloth may reset it
    torch._dynamo.config.suppress_errors = True
    if hasattr(torch._dynamo.config, "error_on_recompile"):
        torch._dynamo.config.error_on_recompile = False

    adapter_dir = (
        args.adapter if os.path.isdir(args.adapter)
        else snapshot_download(repo_id=args.adapter)
    )
    log.info(f"Attaching adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    # KEY FIX: wrap model.forward with dynamo.disable → forces eager mode
    # for every forward call. Unlike TORCHDYNAMO_DISABLE=1 (which broke
    # Unsloth's import-time patches in v3), this happens AFTER Unsloth has
    # already applied its weight-loading patches, so those stay intact.
    model.forward = torch._dynamo.disable(model.forward)
    log.info("model.forward wrapped with torch._dynamo.disable → pure eager")

    log.info("Loading val dataset...")
    val = load_dataset_from_jsonl("val")
    n_total = len(val)
    n = n_total if args.max_samples < 0 else min(args.max_samples, n_total)
    log.info(f"Val: {n_total} samples, evaluating {n}")

    samples = val[:n]
    y_true, y_pred_raw = [], []
    t_start = time.time()

    for idx, item in enumerate(samples):
        msgs = item["messages"]
        truth = msgs[-1]["content"][0]["text"]
        image = msgs[1]["content"][0]["image"]
        utext = msgs[1]["content"][1]["text"]
        infer = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": utext},
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
                    out = model.generate(
                        **inp, max_new_tokens=16, use_cache=True,
                    )
                gen = tok.decode(
                    out[0][inp["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()
                del inp, out
                break
            except Exception as e:
                log.warning(
                    f"idx={idx} attempt {attempt}: "
                    f"{type(e).__name__}: {str(e)[:200]}"
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
                f"last={gen!r}"
            )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # PREFIX normalize (longest-first)
    classes = sorted(set(y_true))
    classes_desc = sorted(classes, key=len, reverse=True)
    def normalize(g: str) -> str:
        g = g.strip()
        for c in classes_desc:
            if g.startswith(c):
                return c
        return g

    y_pred = [normalize(g) for g in y_pred_raw]

    # Metrics
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix,
    )
    all_labels = sorted(set(y_true + y_pred))
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, labels=all_labels,
                   average="macro", zero_division=0)
    f1w = f1_score(y_true, y_pred, labels=all_labels,
                   average="weighted", zero_division=0)
    pm = precision_score(y_true, y_pred, labels=all_labels,
                         average="macro", zero_division=0)
    rm = recall_score(y_true, y_pred, labels=all_labels,
                      average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    pp = precision_score(y_true, y_pred, labels=all_labels,
                         average=None, zero_division=0)
    rp = recall_score(y_true, y_pred, labels=all_labels,
                      average=None, zero_division=0)
    fp_ = f1_score(y_true, y_pred, labels=all_labels,
                   average=None, zero_division=0)

    per_class = {
        cls: {
            "precision": float(pp[i]),
            "recall": float(rp[i]),
            "f1": float(fp_[i]),
            "support": int(y_true.count(cls)),
        }
        for i, cls in enumerate(all_labels)
    }

    os.makedirs(args.save_dir, exist_ok=True)

    # Plot CM with Korean-safe font
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

        n_lbl = len(all_labels)
        fs = max(10, n_lbl * 0.5)
        fig, ax = plt.subplots(figsize=(fs, fs))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(n_lbl))
        ax.set_yticks(range(n_lbl))
        short = [l[:8] for l in all_labels]
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(short, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(
            f"Confusion Matrix (v6)\n"
            f"Acc={acc:.3f}  F1_macro={f1m:.3f}  "
            f"F1_weighted={f1w:.3f}  n={len(y_true)}"
        )
        thr = cm.max() / 2
        for i in range(n_lbl):
            for j in range(n_lbl):
                if cm[i, j] > 0:
                    ax.text(
                        j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > thr else "black",
                    )
        plt.tight_layout()
        cm_path = os.path.join(args.save_dir, f"confusion_matrix_{args.label}.png")
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        log.info(f"CM saved: {cm_path}")
    except Exception as e:
        log.warning(f"CM plot failed: {e}")

    # Save metrics + raw preds
    result = {
        "accuracy": acc,
        "precision_macro": pm, "recall_macro": rm,
        "f1_macro": f1m, "f1_weighted": f1w,
        "per_class": per_class,
        "labels": all_labels,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_path": cm_path,
        "total": len(y_true),
    }
    with open(os.path.join(args.save_dir, "metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str, ensure_ascii=False)
    with open(os.path.join(args.save_dir, "raw_predictions.json"), "w",
              encoding="utf-8") as f:
        json.dump(
            [{"truth": t, "raw": r, "normalized": n}
             for t, r, n in zip(y_true, y_pred_raw, y_pred)],
            f, indent=2, ensure_ascii=False,
        )

    # Summary
    total_time = time.time() - t_start
    print("=" * 70)
    print(f"Adapter:    {args.adapter}")
    print(f"Base:       {args.base}")
    print(f"Samples:    {result['total']}   (elapsed {total_time/60:.1f} min)")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {pm:.4f}")
    print(f"Recall:     {rm:.4f}")
    print(f"F1 macro:   {f1m:.4f}")
    print(f"F1 weighted:{f1w:.4f}")
    print(f"CM PNG:     {cm_path}")
    print(f"Metrics:    {os.path.join(args.save_dir, 'metrics.json')}")
    print(f"Raw preds:  {os.path.join(args.save_dir, 'raw_predictions.json')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
