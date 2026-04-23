#!/usr/bin/env python3
"""V8 eval — fully uniform shapes = zero recompile drops target.

v7 reduced drops from 50% to 2.4% by fixing image size + prompt. Remaining
variance comes from the generation loop: different predictions have
different output lengths, causing KV cache shape variation across calls.

v8 fix: force exact output length via min_new_tokens=max_new_tokens=16.
Every generation emits exactly 16 tokens; KV cache shape is identical
for every call. One compile cache entry = zero limit hits = zero drops.

PREFIX normalization still rescues the valid pest name from the first few
tokens, ignoring the trailing chat/padding tokens.
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
FIXED_OUTPUT_TOKENS = 16   # exact min = exact max = uniform KV cache shape


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
    ap.add_argument("--adapter", default="pfox1995/pest-detector-final")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--save-dir", default="/workspace/hf_eval_v8")
    ap.add_argument("--label", default="v8")
    ap.add_argument("--max-samples", type=int, default=-1)
    args = ap.parse_args()

    from unsloth import FastVisionModel
    from peft import PeftModel
    from huggingface_hub import snapshot_download

    log.info(f"Loading base: {args.base}")
    model, tok = FastVisionModel.from_pretrained(args.base, load_in_4bit=False)

    torch._dynamo.config.suppress_errors = True
    for attr in ("error_on_recompile", "fail_on_recompile_limit_hit"):
        if hasattr(torch._dynamo.config, attr):
            setattr(torch._dynamo.config, attr, False)

    adapter_dir = (
        args.adapter if os.path.isdir(args.adapter)
        else snapshot_download(repo_id=args.adapter)
    )
    log.info(f"Attaching adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    # Force eager forward
    model.forward = torch._dynamo.disable(model.forward)
    log.info("model.forward wrapped with torch._dynamo.disable → pure eager")

    log.info("Materializing val samples (letterbox 512x512, fixed prompt)...")
    t0 = time.time()
    all_samples = list(iter_val_samples())
    if args.max_samples > 0:
        all_samples = all_samples[: args.max_samples]
    n = len(all_samples)
    log.info(f"Val samples prepared: {n} (took {time.time()-t0:.1f}s)")

    # Get the tokenizer's pad_token_id (needed for fixed-length generation)
    pad_id = (
        tok.tokenizer.pad_token_id
        if hasattr(tok, "tokenizer") and tok.tokenizer.pad_token_id is not None
        else (tok.tokenizer.eos_token_id if hasattr(tok, "tokenizer") else None)
    )
    log.info(f"pad_token_id: {pad_id}")

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
                        max_new_tokens=FIXED_OUTPUT_TOKENS,
                        min_new_tokens=FIXED_OUTPUT_TOKENS,  # force exact length
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

    classes = sorted(set(y_true))
    classes_desc = sorted(classes, key=len, reverse=True)

    def normalize(g: str) -> str:
        g = g.strip()
        for c in classes_desc:
            if g.startswith(c):
                return c
        return g

    y_pred = [normalize(g) for g in y_pred_raw]

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
        ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(all_labels, fontsize=8)
        ax.set_xlabel("예측 (Predicted)")
        ax.set_ylabel("실제 (Actual)")
        ax.set_title(
            f"Confusion Matrix (v8 — 1595 samples, fixed shapes + fixed output)\n"
            f"Acc={acc:.4f}  F1_macro={f1m:.4f}  "
            f"F1_weighted={f1w:.4f}  n={len(y_true)}"
        )
        thr = cm.max() / 2
        for i in range(n_lbl):
            for j in range(n_lbl):
                if cm[i, j] > 0:
                    ax.text(
                        j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > thr else "black",
                    )
        plt.tight_layout()
        cm_path = os.path.join(args.save_dir, f"confusion_matrix_{args.label}.png")
        fig.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"CM saved: {cm_path}")
    except Exception as e:
        log.warning(f"CM plot failed: {e}")

    result = {
        "accuracy": acc,
        "precision_macro": pm, "recall_macro": rm,
        "f1_macro": f1m, "f1_weighted": f1w,
        "per_class": per_class,
        "labels": all_labels,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_path": cm_path,
        "total": len(y_true),
        "config": {
            "image_size": FIXED_SIZE,
            "prompt": FIXED_PROMPT,
            "output_tokens": FIXED_OUTPUT_TOKENS,
            "adapter": args.adapter,
            "base": args.base,
        },
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

    total_time = time.time() - t_start
    print("=" * 70)
    print(f"Adapter:    {args.adapter}")
    print(f"Base:       {args.base}")
    print(f"Preprocess: letterbox {FIXED_SIZE}x{FIXED_SIZE}, fixed prompt, fixed output={FIXED_OUTPUT_TOKENS}")
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
