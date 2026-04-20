#!/usr/bin/env python3
"""
Upload the retrained LoRA adapter + evaluation to HuggingFace.
================================================================
Run after `hp_search.py --retrain` produces the best model at
`${HP_BEST_MODEL_DIR}/lora/`.

Reads from env:
    HF_TOKEN          HuggingFace token with write access (required)
    HF_REPO_ID        Target model repo (default: pfox1995/qwen35-9b-pest-detector-lora)
    HF_PRIVATE        "true"/"false" (default: true)
    HP_BEST_MODEL_DIR Local folder containing lora/ + evaluation/ (default: /workspace/best-pest-detector)

Usage:
    python hf_upload.py
"""

import datetime
import json
import os
import sys

from huggingface_hub import create_repo, login, upload_file, upload_folder


def main():
    token    = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set.")
        sys.exit(1)

    repo_id  = os.environ.get("HF_REPO_ID", "pfox1995/qwen35-9b-pest-detector-lora")
    private  = os.environ.get("HF_PRIVATE", "true").lower() == "true"
    best_dir = os.environ.get("HP_BEST_MODEL_DIR", "/workspace/best-pest-detector")

    lora_path = os.path.join(best_dir, "lora")
    eval_dir  = os.path.join(best_dir, "evaluation")
    metrics_path = os.path.join(eval_dir, "metrics.json")

    if not os.path.isdir(lora_path):
        print(f"ERROR: LoRA adapter folder not found: {lora_path}")
        sys.exit(1)

    login(token=token, add_to_git_credential=False)

    print(f"대상: https://huggingface.co/{repo_id}  (private={private})")
    create_repo(repo_id, private=private, repo_type="model", exist_ok=True)

    # ─── Build model card ──────────────────────────────────────────────
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    readme = MODEL_CARD_TEMPLATE.format(
        ts=ts,
        acc=metrics.get("accuracy", 0),
        f1m=metrics.get("f1_macro", 0),
        f1w=metrics.get("f1_weighted", 0),
        prec=metrics.get("precision_macro", 0),
        rec=metrics.get("recall_macro", 0),
        repo=repo_id,
    )
    readme_path = "/tmp/hf_readme.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)

    # ─── Upload adapter ────────────────────────────────────────────────
    print(f"어댑터 폴더 업로드 중: {lora_path}")
    upload_folder(
        folder_path=lora_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload LoRA adapter ({ts})",
    )

    # ─── Upload evaluation artifacts ───────────────────────────────────
    if os.path.isdir(eval_dir):
        print(f"평가 결과 업로드 중: {eval_dir}")
        upload_folder(
            folder_path=eval_dir,
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="evaluation",
            commit_message=f"Upload evaluation artifacts ({ts})",
        )

    # ─── Upload README last (contains filled-in metrics) ───────────────
    upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Update model card ({ts})",
    )

    print(f"완료: https://huggingface.co/{repo_id}")


MODEL_CARD_TEMPLATE = """---
base_model: unsloth/Qwen3.5-9B
tags:
  - vision-language
  - pest-detection
  - lora
  - korean
  - qwen
  - unsloth
library_name: peft
pipeline_tag: image-text-to-text
---

# 해충 탐지 LoRA 어댑터 (Pest Detection)

Qwen3.5-9B 기반 19종 해충 이미지 분류 LoRA 어댑터.
AI Hub 한국어 해충 이미지 데이터셋으로 학습됨.

**학습 완료**: {ts}

## 평가 성능 (Hold-out val set, 500 샘플)

| 지표 | 값 |
|:---|:---|
| Accuracy          | {acc:.4f} |
| F1 (macro)        | {f1m:.4f} |
| F1 (weighted)     | {f1w:.4f} |
| Precision (macro) | {prec:.4f} |
| Recall (macro)    | {rec:.4f} |

## 사용법

```python
from unsloth import FastVisionModel
from peft import PeftModel

base, tok = FastVisionModel.from_pretrained(
    'unsloth/Qwen3.5-9B',
    load_in_4bit=True,
)
model = PeftModel.from_pretrained(base, '{repo}')
FastVisionModel.for_inference(model)
```

## 학습 상세

- Base model: [unsloth/Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B)
- Framework: Unsloth + TRL + PEFT
- Search: Optuna 2-phase (50 proxy + 10 full trials, TPE sampler)
- 학습 코드: https://github.com/pfox1995/pest-hyperparameter-search
"""


if __name__ == "__main__":
    main()
