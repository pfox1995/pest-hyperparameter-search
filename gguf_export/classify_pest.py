#!/usr/bin/env python3
"""
Classify a pest image via the local llama-server (Q5_K_M GGUF).

Uses three production-grade reliability levers:

1. **Thinking-mode disabled**: training was non-thinking. We send
   `chat_template_kwargs.enable_thinking=false` and `reasoning_format=none`
   per request, defending against the llama.cpp Qwen3.5 default-on bug
   (ggml-org/llama.cpp#20833 / #20409). The server should also be launched
   with `--reasoning off --reasoning-format none --reasoning-budget 0`.

2. **Class-aware system prompt**: explicitly enumerates the 19 trained
   classes. Helps the un-fine-tuned residual generation pick from the
   right vocabulary, even on the LoRA-zeroed Gated DeltaNet path.

3. **GBNF grammar constraint**: forces the next-token sampler to ONLY pick
   tokens that complete one of the 19 class strings. Eliminates "echo
   prompt", Chinese fallback, "wrong-but-similar-class" Hangul corruptions.
   See gguf_export/pest_classes.gbnf.

Env:
    SERVER_URL     default: http://localhost:8080
    MAX_TOKENS     default: 30 (Korean class names are ≤ 8 tokens, but
                                 we leave headroom in case grammar pushes
                                 longer alternates)

Usage:
    python classify_pest.py /path/to/image.jpg
    python classify_pest.py /workspace/val_samples/val/*/*.jpg --bench
"""

import argparse
import base64
import glob
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

PEST_CLASSES = [
    "검거세미밤나방", "꽃노랑총채벌레", "담배가루이", "담배거세미나방",
    "담배나방", "도둑나방", "먹노린재", "목화바둑명나방", "무잎벌",
    "배추좀나방", "배추흰나비", "벼룩잎벌레", "비단노린재", "썩덩나무노린재",
    "알락수염노린재", "정상", "큰28점박이무당벌레", "톱다리개미허리노린재",
    "파밤나방",
]

SYSTEM_PROMPT = (
    "당신은 작물 해충 식별 전문가입니다. "
    "사진을 보고 다음 19개의 분류 중 하나만 한국어로 답하세요:\n"
    + ", ".join(PEST_CLASSES)
    + '\n해충이 없으면 "정상"이라고 답하세요. '
    "다른 어떤 텍스트나 설명도 포함하지 마세요."
)
USER_PROMPT = "이 사진에 있는 해충의 이름을 알려주세요."

GRAMMAR = (
    'root ::= '
    + ' | '.join(f'"{c}"' for c in PEST_CLASSES)
)


def classify(image_path: str, server_url: str, max_tokens: int = 30) -> dict:
    img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ],
        "temperature": 0.0,
        "top_k": 1,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_format": "none",
        "grammar": GRAMMAR,
    }
    req = urllib.request.Request(
        f"{server_url}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r:
        body = json.loads(r.read())
    elapsed = time.time() - t0
    msg = body["choices"][0]["message"]
    raw = (msg.get("content") or "").strip()
    # The chat template prepends `<think>\n\n</think>\n\n` even when thinking
    # is disabled (llama.cpp Qwen3.5 quirk). The grammar guarantees the actual
    # answer is one of PEST_CLASSES somewhere in the string — find it.
    pred = raw
    for cls in PEST_CLASSES:
        if cls in raw:
            pred = cls
            break
    return {"pred": pred, "elapsed_s": elapsed, "raw_content": raw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Image file(s) or glob(s).")
    ap.add_argument("--server", default=os.environ.get("SERVER_URL", "http://localhost:8080"))
    ap.add_argument("--max-tokens", type=int, default=30)
    ap.add_argument("--bench", action="store_true",
                    help="Treat parent dir as ground-truth label; print accuracy.")
    args = ap.parse_args()

    files: list[str] = []
    for p in args.paths:
        files.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        sys.exit("no input files")

    correct = 0
    per_class: dict[str, list[int]] = {}
    for f in files:
        truth = Path(f).parent.name if args.bench else None
        out = classify(f, args.server, args.max_tokens)
        ok = (truth and out["pred"] == truth)
        if truth:
            per_class.setdefault(truth, [0, 0])
            per_class[truth][0] += int(ok)
            per_class[truth][1] += 1
            correct += int(ok)
        marker = ("✓" if ok else "✗") if truth else " "
        print(f"{marker} pred={out['pred']:<20s} ({out['elapsed_s']:.1f}s){' truth=' + truth if truth else ''}  [{Path(f).name}]")

    if args.bench and per_class:
        total = sum(t for _, t in per_class.values())
        print(f"\n=== ACCURACY: {correct}/{total} = {100 * correct / total:.1f}% ===")
        for cls, (c, t) in sorted(per_class.items(), key=lambda x: -x[1][0] / max(1, x[1][1])):
            print(f"  {c}/{t}  {100 * c / t:5.1f}%  {cls}")


if __name__ == "__main__":
    main()
