#!/usr/bin/env python3
"""
Update only the README.md on the HuggingFace Hub for
pfox1995/pest-detector-final, using the polished Korean
version in README_hf_korean_complete.md.

Usage (Windows PowerShell):
    $env:HF_TOKEN = "hf_xxx..."          # your HF write token
    python hf_update_readme.py

Usage (bash):
    HF_TOKEN=hf_xxx... python hf_update_readme.py
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID   = os.environ.get("HF_REPO_ID", "pfox1995/pest-detector-final")
LOCAL_MD  = Path(__file__).parent / "README_hf_korean_complete.md"


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN 환경변수가 설정되어 있지 않습니다.", file=sys.stderr)
        print("       PowerShell:  $env:HF_TOKEN = 'hf_xxx...'", file=sys.stderr)
        print("       bash:        export HF_TOKEN=hf_xxx...",   file=sys.stderr)
        return 1

    if not LOCAL_MD.is_file():
        print(f"ERROR: 소스 파일을 찾을 수 없습니다: {LOCAL_MD}", file=sys.stderr)
        return 1

    size_kb = LOCAL_MD.stat().st_size / 1024
    print(f"업로드 대상:  https://huggingface.co/{REPO_ID}/blob/main/README.md")
    print(f"로컬 파일:    {LOCAL_MD}  ({size_kb:.1f} KB)")
    print("업로드 시작…")

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(LOCAL_MD),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Add training stability & checkpoint-selection section (Korean)",
    )

    print(f"✅ 완료: https://huggingface.co/{REPO_ID}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
