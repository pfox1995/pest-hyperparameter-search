#!/usr/bin/env python3
"""
Robust HuggingFace dataset downloader with rate-limit handling.
===============================================================
Handles the 19.4GB pest-detection-korean dataset gracefully:
  - Uses hf_transfer (Rust parallel downloader) when available
  - Retries with exponential backoff on HTTP 429 rate limits
  - Resumes interrupted downloads (skips already-downloaded files)
  - Logs progress per-subfolder so you can see movement
  - Verifies completion before writing the sentinel file

Usage:
    python download_dataset.py              # Normal download
    python download_dataset.py --verify     # Verify existing download only
"""

import os
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ID    = "Himedia-AI-01/pest-detection-korean"
DATA_DIR   = os.environ.get("HP_DATA_DIR", "/workspace/data")
TOKEN      = os.environ.get("HF_TOKEN") or None
SENTINEL   = os.path.join(DATA_DIR, ".download_complete")

# Required files to consider the download complete
REQUIRED_FILES = ["train.jsonl", "val.jsonl"]

MAX_RETRIES    = 5
INITIAL_BACKOFF = 30   # seconds — HF rate limits usually lift in 30-60s


def check_hf_transfer():
    """Check if hf_transfer is installed and enabled."""
    enabled = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") == "1"
    try:
        import hf_transfer  # noqa: F401
        installed = True
    except ImportError:
        installed = False

    if installed and enabled:
        logger.info("hf_transfer 활성화됨 (Rust 병렬 다운로더)")
    elif installed and not enabled:
        logger.info("hf_transfer 설치됨 — HF_HUB_ENABLE_HF_TRANSFER=1로 활성화하세요")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        logger.info("자동 활성화 완료")
    else:
        logger.info("hf_transfer 미설치 — 기본 다운로더 사용 (느림)")


def download_with_retry():
    """Download dataset with retry + exponential backoff on rate limits."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    os.makedirs(DATA_DIR, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                f"다운로드 시도 {attempt}/{MAX_RETRIES}: "
                f"{REPO_ID} → {DATA_DIR}"
            )

            snapshot_download(
                REPO_ID,
                repo_type="dataset",
                local_dir=DATA_DIR,
                local_dir_use_symlinks=False,
                token=TOKEN,
                # Force re-check of incomplete files
                force_download=False,
                # Increase per-file timeout for large images
                etag_timeout=30,
            )

            logger.info("snapshot_download 완료")
            return True

        except HfHubHTTPError as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                backoff = INITIAL_BACKOFF * (2 ** (attempt - 1))
                # Cap at 5 minutes
                backoff = min(backoff, 300)
                logger.warning(
                    f"Rate limit (429) — {backoff}초 대기 후 재시도 "
                    f"({attempt}/{MAX_RETRIES})"
                )
                time.sleep(backoff)
            else:
                logger.error(f"HF HTTP 오류: {e}")
                if attempt < MAX_RETRIES:
                    wait = 10 * attempt
                    logger.info(f"{wait}초 후 재시도...")
                    time.sleep(wait)
                else:
                    raise

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"네트워크 오류: {e}")
            if attempt < MAX_RETRIES:
                wait = 15 * attempt
                logger.info(f"{wait}초 후 재시도...")
                time.sleep(wait)
            else:
                raise

        except Exception as e:
            # Catch-all: some HF errors wrap rate limits in generic exceptions
            error_str = str(e).lower()
            if "429" in str(e) or "rate" in error_str or "too many" in error_str:
                backoff = INITIAL_BACKOFF * (2 ** (attempt - 1))
                backoff = min(backoff, 300)
                logger.warning(
                    f"Rate limit 감지 — {backoff}초 대기 후 재시도 "
                    f"({attempt}/{MAX_RETRIES})"
                )
                time.sleep(backoff)
            else:
                logger.error(f"예상치 못한 오류: {e}")
                raise

    logger.error(f"{MAX_RETRIES}회 시도 후 다운로드 실패")
    return False


def verify_download():
    """Verify that all required files exist and dataset looks complete."""
    missing = []
    for fname in REQUIRED_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            missing.append(fname)

    if missing:
        logger.error(f"필수 파일 누락: {missing}")
        return False

    # Count files and total size
    total_files = 0
    total_bytes = 0
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            fpath = os.path.join(root, f)
            total_files += 1
            total_bytes += os.path.getsize(fpath)

    total_gb = total_bytes / 1024**3
    logger.info(
        f"데이터셋 검증 완료: {total_files:,}개 파일, {total_gb:.1f}GB"
    )

    # Sanity check — dataset should be at least 15GB
    if total_gb < 15.0:
        logger.warning(
            f"데이터셋이 예상보다 작습니다 ({total_gb:.1f}GB < 15GB). "
            f"다운로드가 불완전할 수 있습니다."
        )
        return False

    # Count samples in JSONL files
    for fname in REQUIRED_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        with open(fpath, "r") as f:
            n_lines = sum(1 for _ in f)
        logger.info(f"  {fname}: {n_lines:,}개 샘플")

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing download only")
    args = parser.parse_args()

    if args.verify:
        ok = verify_download()
        sys.exit(0 if ok else 1)

    # Skip if already downloaded
    if os.path.exists(SENTINEL):
        logger.info("다운로드 완료 마커 존재 — 건너뜁니다.")
        if verify_download():
            return
        else:
            logger.warning("마커는 있지만 검증 실패 — 재다운로드합니다.")
            os.remove(SENTINEL)

    check_hf_transfer()

    logger.info(f"{'='*60}")
    logger.info(f"  HuggingFace 데이터셋 다운로드")
    logger.info(f"  Repo: {REPO_ID}")
    logger.info(f"  경로: {DATA_DIR}")
    logger.info(f"  예상 크기: ~19.4GB")
    logger.info(f"{'='*60}")

    start = time.time()
    success = download_with_retry()

    if not success:
        logger.error("다운로드 실패")
        sys.exit(1)

    elapsed = (time.time() - start) / 60
    logger.info(f"다운로드 소요 시간: {elapsed:.1f}분")

    # Verify
    if verify_download():
        # Write sentinel
        with open(SENTINEL, "w") as f:
            f.write(f"downloaded at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"elapsed: {elapsed:.1f} min\n")
        logger.info("다운로드 + 검증 완료!")
    else:
        logger.error("다운로드는 완료되었으나 검증 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()
