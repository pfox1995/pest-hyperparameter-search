#!/usr/bin/env python3
"""
Seed a fresh Optuna study with the top N trials from an existing DB.
=====================================================================
Use when you want to preserve known-good params but reset the study —
e.g. after changing the proxy compute budget, old eval_loss values are
no longer comparable to new ones, but the param choices still transfer.

Usage (on RunPod):
    python seed_from_db.py --top-n 3              # Backup + wipe + seed
    python seed_from_db.py --top-n 3 --dry-run    # Preview only
    python seed_from_db.py --source <path>        # Read from custom DB
"""

import argparse
import os
import shutil
import sys

import optuna
from optuna.samplers import TPESampler

VOLUME_DIR  = os.environ.get("HP_VOLUME_DIR", "/workspace")
DEFAULT_DB  = os.environ.get("HP_DB_PATH", f"{VOLUME_DIR}/hp_search_results.db")
STUDY_NAME  = "pest-detection-hpsearch"
RANDOM_SEED = 42


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--source", default=DEFAULT_DB,
                   help="DB to read trials from (default: current DB)")
    p.add_argument("--target", default=DEFAULT_DB,
                   help="DB to write fresh study to (default: same as source)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-backup", action="store_true")
    args = p.parse_args()

    if not os.path.exists(args.source):
        print(f"Source DB not found: {args.source}")
        sys.exit(1)

    # ─── Read top N from source (list is fully materialized) ──────────
    old_study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{args.source}",
    )
    completed = [t for t in old_study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value is not None]
    if not completed:
        print("No completed trials to seed from.")
        sys.exit(1)

    completed.sort(key=lambda t: t.value)
    top = completed[:args.top_n]

    print(f"\nTop {len(top)} trials from {args.source}:\n")
    for t in top:
        print(f"  #{t.number:3d}  eval_loss={t.value:.6f}")
        for k, v in sorted(t.params.items()):
            print(f"       {k:22s} = {v}")
        print()

    if args.dry_run:
        print("[DRY RUN] No writes.")
        return

    # ─── Backup + wipe target ─────────────────────────────────────────
    if os.path.exists(args.target):
        if not args.no_backup:
            backup = args.target + ".bak"
            shutil.copy2(args.target, backup)
            print(f"Backed up existing DB -> {backup}")
        os.remove(args.target)
        print(f"Removed: {args.target}")

    # ─── Fresh study + enqueue ────────────────────────────────────────
    new_study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{args.target}",
        direction="minimize",
        sampler=TPESampler(
            seed=RANDOM_SEED, n_startup_trials=3, multivariate=True,
        ),
    )
    for t in top:
        new_study.enqueue_trial(t.params)
        print(f"Enqueued old trial #{t.number} (eval_loss={t.value:.6f})")

    print(f"\nDone. Fresh study seeded with {len(top)} trials at {args.target}")
    print("Next: python hp_search.py --proxy --n-trials 50")


if __name__ == "__main__":
    main()
