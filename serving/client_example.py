#!/usr/bin/env python3
"""Minimal Python client for the pest-detector serving endpoint.

Usage:
    # single image
    python client_example.py --url http://POD:8000 --image pest.jpg

    # folder (uses /predict_batch — up to 16 at a time)
    python client_example.py --url http://POD:8000 --dir ./samples
"""
import argparse
import base64
import glob
import json
import os
import sys
import time

import requests


def b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def predict_one(url: str, path: str):
    r = requests.post(
        f"{url}/predict",
        json={"image_base64": b64(path)},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def predict_batch(url: str, paths: list[str]):
    r = requests.post(
        f"{url}/predict_batch",
        json={"images_base64": [b64(p) for p in paths]},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["results"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000",
                    help="Base URL of the serving endpoint")
    ap.add_argument("--image", help="Single image file to classify")
    ap.add_argument("--dir", help="Folder of images — classify all in one batch")
    args = ap.parse_args()

    # Health check first
    try:
        h = requests.get(f"{args.url}/health", timeout=5).json()
        print(f"health: {json.dumps(h, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"[error] cannot reach {args.url}: {e}", file=sys.stderr)
        sys.exit(1)

    if args.image:
        t0 = time.time()
        res = predict_one(args.url, args.image)
        elapsed = time.time() - t0
        print(f"\n{args.image}")
        print(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"wall: {elapsed*1000:.0f}ms  (server: {res['latency_ms']}ms)")

    elif args.dir:
        paths = sorted(
            glob.glob(os.path.join(args.dir, "*.jpg"))
            + glob.glob(os.path.join(args.dir, "*.jpeg"))
            + glob.glob(os.path.join(args.dir, "*.png"))
        )
        if not paths:
            print(f"[error] no images found in {args.dir}", file=sys.stderr)
            sys.exit(1)

        print(f"classifying {len(paths)} images in batches of 16")
        for i in range(0, len(paths), 16):
            chunk = paths[i:i+16]
            t0 = time.time()
            results = predict_batch(args.url, chunk)
            elapsed = time.time() - t0
            for p, r in zip(chunk, results):
                print(f"  {os.path.basename(p):40s} → {r['pest']:20s} "
                      f"({r['latency_ms']}ms, known={r['known_class']})")
            print(f"  batch of {len(chunk)}: {elapsed:.2f}s total\n")
    else:
        ap.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
