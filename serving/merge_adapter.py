#!/usr/bin/env python3
"""Merge LoRA adapter into base model for production serving.

Outputs a single directory containing:
  - merged bf16 model weights (no PEFT wrapper)
  - tokenizer / processor config
  - classes.json (extracted from eval metrics.json if available)

After this, `serve.py` can load the directory with plain transformers,
no peft / unsloth dependency at serve time.

Usage:
    python merge_adapter.py                              # defaults
    python merge_adapter.py --out /path/to/merged
    python merge_adapter.py --adapter user/my-adapter
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel


DEFAULT_BASE    = "unsloth/Qwen3.5-9B"
DEFAULT_ADAPTER = "pfox1995/pest-detector-final"
DEFAULT_OUT     = "/workspace/pest-detector-merged"

# Fallback class list — the 19 Korean pest labels plus "정상" (normal).
# Used if we can't find an eval metrics.json to extract from.
# Re-derive from val.jsonl if you're unsure this is current for your adapter.
FALLBACK_CLASSES = [
    "정상",
    "벼룩잎벌레",
    "썩덩나무노린재",
    "톱다리개미허리노린재",
    "파밤나방",
    "알락수염노린재",
    "배추흰나비",
    "먹노린재",
    "배추좀나방",
    "담배나방",
    "꽃노랑총채벌레",
    "목화바둑명나방",
    "담배가루이",
    "비단노린재",
    "큰28점박이무당벌레",
    "담배거세미나방",
    "배추바둑나방",
    "복숭아혹진딧물",
    "도둑나방",
    "큰28점박이무당벌레",
]


def find_classes() -> list[str]:
    """Try to populate CLASSES from any recent eval metrics.json on disk."""
    candidates = [
        "/workspace/hf_eval_v3/metrics.json",
        "/workspace/hf_eval_v2/metrics.json",
        "/workspace/hf_eval/metrics.json",
        "/workspace/golden_eval/metrics.json",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    m = json.load(f)
                labels = m.get("labels") or list(m.get("per_class", {}).keys())
                if labels:
                    print(f"[classes] populated {len(labels)} from {p}")
                    return sorted(set(labels))
            except Exception as e:
                print(f"[classes] skipping {p} ({e})")
    print(
        f"[classes] no eval metrics.json found — using fallback "
        f"({len(FALLBACK_CLASSES)} classes). VERIFY against your training labels."
    )
    return sorted(set(FALLBACK_CLASSES))


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda",
                    help="cuda / cpu — cpu works but needs ~40 GB RAM")
    args = ap.parse_args()

    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[warn] {out_dir} already exists and is non-empty; "
              f"will overwrite safetensors files.")

    print(f"[1/4] Loading base model: {args.base} (bf16)")
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map=args.device,
    )

    print(f"[2/4] Attaching adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("[3/4] Merging LoRA into base weights (this may take ~30 s)")
    model = model.merge_and_unload()

    print(f"[4/4] Saving merged model → {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")

    processor = AutoProcessor.from_pretrained(args.base)
    processor.save_pretrained(out_dir)

    # classes.json for PREFIX normalization at serve time
    classes = find_classes()
    with open(out_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    print(f"[classes] wrote {len(classes)} labels to {out_dir}/classes.json")

    total_bytes = sum(
        p.stat().st_size for p in out_dir.rglob("*") if p.is_file()
    )
    print(f"\n[done] merged model: {total_bytes / 1024**3:.1f} GB")
    print(f"[done] start server:   uvicorn serve:app --host 0.0.0.0 --port 8000")
    print(f"[done] env:            MODEL_DIR={out_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupted]", file=sys.stderr)
        sys.exit(130)
