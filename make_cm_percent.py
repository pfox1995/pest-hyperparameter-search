#!/usr/bin/env python3
"""
Render row-normalized percentage confusion matrices from any metrics
JSON file produced by this project.

Each row is divided by its row sum so every row sums to 100%. The
diagonal values then read directly as per-class recall.

Usage:
    # Single file
    python make_cm_percent.py \
        --json evaluation/metrics_final.json \
        --out  evaluation/confusion_matrix_final_percent.png \
        --title "pfox1995/pest-detector-final"

    # All three CMs used by the HF card
    python make_cm_percent.py --all
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


HERE = Path(__file__).parent
EVAL = HERE / "evaluation"


def find_korean_font() -> str | None:
    """Pick the first Korean-capable font.

    Ordered from most to least preferred. MS Gothic (Japanese) has no
    Hangul glyphs, so we explicitly prefer fonts known to carry Korean
    syllables. Malgun Gothic is guaranteed on default Windows installs.
    """
    preferred = [
        "malgun",        # Windows default Korean — best
        "nanumgothic",   # popular Korean web font
        "nanum",         # NanumBarunGothic, NanumMyeongjo, ...
        "notosanskr",    # Noto Sans KR
        "notosanscjkkr",
        "batang",        # Batang / BatangChe
        "gulim",         # Gulim / GulimChe
        "dotum",
        "noto cjk",      # generic CJK Noto fallback
    ]
    fonts = font_manager.findSystemFonts()
    if not fonts:
        font_manager._load_fontmanager(try_read_cache=False)
        fonts = font_manager.findSystemFonts()
    lowered = [(p, os.path.basename(p).lower()) for p in fonts]
    for key in preferred:
        for path, name in lowered:
            if key in name:
                return path
    return None


def render_cm(json_path: Path, out_path: Path, subtitle: str) -> None:
    data   = json.loads(json_path.read_text(encoding="utf-8"))
    labels = data["labels"]
    cm     = np.asarray(data["confusion_matrix"], dtype=np.float64)
    acc    = data["accuracy"]
    f1m    = data["f1_macro"]
    f1w    = data["f1_weighted"]
    total  = int(cm.sum())

    # Row-normalize to percentages. Rows with zero support stay zero.
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    cm_pct = cm / row_sum * 100.0

    font_path = find_korean_font()
    if font_path:
        plt.rcParams["font.family"] = (
            font_manager.FontProperties(fname=font_path).get_name()
        )
    else:
        print("WARN: no Korean font found; labels may render as tofu.",
              file=sys.stderr)
    plt.rcParams["axes.unicode_minus"] = False

    n = len(labels)
    # Give slightly more space when UNKNOWN is in the matrix (n=20)
    # because the label is wider than the pest names.
    fig, ax = plt.subplots(figsize=(max(11, n * 0.62), max(10, n * 0.58)))
    im = ax.imshow(cm_pct, interpolation="nearest",
                   cmap="Blues", vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, format="%d%%")
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Row-normalized %", fontsize=10)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("예측 (Predicted)", fontsize=11)
    ax.set_ylabel("실제 (Actual)",   fontsize=11)
    ax.set_title(
        f"Confusion Matrix (%) — {subtitle}\n"
        f"Row-normalized to recall.  "
        f"Acc={acc*100:.2f}%  F1_macro={f1m*100:.2f}%  "
        f"F1_weighted={f1w*100:.2f}%  n={total}",
        fontsize=11, pad=12,
    )

    # Annotate non-zero cells. Diagonal cells get bold weight so
    # per-class recall pops out.
    for i in range(n):
        for j in range(n):
            v = cm_pct[i, j]
            if v < 0.5:
                continue
            text = f"{v:.0f}" if v >= 10 else f"{v:.1f}"
            ax.text(
                j, i, text,
                ha="center", va="center",
                fontsize=8 if v >= 10 else 7,
                color="white" if v >= 50 else "black",
                fontweight="bold" if i == j else "normal",
            )

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    kb = out_path.stat().st_size / 1024
    print(f"[OK] {out_path.name}  ({kb:.1f} KB)")


# The three CMs referenced by the HF card README.
BATCH = [
    # (metrics JSON,                            output PNG,                                        title)
    ("metrics_final.json",                      "confusion_matrix_final_percent.png",              "pfox1995/pest-detector-final (19 classes)"),
    ("metrics_final_with_unknown.json",         "confusion_matrix_final_with_unknown_percent.png", "pfox1995/pest-detector-final (19 + UNKNOWN)"),
    ("metrics_baseline_final.json",             "confusion_matrix_baseline_final_percent.png",     "Baseline Qwen3.5-9B, no LoRA (19 + UNKNOWN)"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",  type=Path, help="Single metrics JSON path")
    ap.add_argument("--out",   type=Path, help="Output PNG path")
    ap.add_argument("--title", default="",  help="Subtitle for plot")
    ap.add_argument("--all",   action="store_true",
                    help="Render the 3 CMs used by the HF card")
    args = ap.parse_args()

    if args.all:
        for j, o, t in BATCH:
            render_cm(EVAL / j, EVAL / o, t)
        return 0

    if not (args.json and args.out):
        ap.error("--json and --out are required unless --all is used")
    render_cm(args.json, args.out, args.title or args.json.stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
