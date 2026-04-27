#!/usr/bin/env python3
"""
Replace the 3 confusion-matrix PNGs on the HF model card with their
percentage-normalized counterparts. Reads HF_TOKEN from the environment
only — the token is never written to disk.

Usage:
    $env:HF_TOKEN = "hf_..."
    python hf_upload_cms.py
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = os.environ.get("HF_REPO_ID", "pfox1995/pest-detector-final")
EVAL    = Path(__file__).parent / "evaluation"

# (local filename, path_in_repo on HF)
FILES = [
    # The 19-class CM — README refers to this as plain confusion_matrix.png
    ("confusion_matrix_final_percent.png",              "confusion_matrix.png"),
    # 20x20 LoRA CM with UNKNOWN column
    ("confusion_matrix_final_with_unknown_percent.png", "confusion_matrix_final_with_unknown.png"),
    # 20x20 baseline CM with UNKNOWN column
    ("confusion_matrix_baseline_final_percent.png",     "confusion_matrix_baseline_final.png"),
]


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN env var not set.", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    for local, remote in FILES:
        local_path = EVAL / local
        if not local_path.is_file():
            print(f"ERROR: missing {local_path}", file=sys.stderr)
            return 1
        kb = local_path.stat().st_size / 1024
        print(f"Uploading {local}  ({kb:.1f} KB)  -> {REPO_ID}/{remote}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=f"Replace {remote} with row-normalized percentage version",
        )
    print(f"[OK] https://huggingface.co/{REPO_ID}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
