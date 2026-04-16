#!/usr/bin/env python3
"""
Seed an Optuna study with the best trials from W&B.
=========================================================
Run this on a new pod BEFORE hp_search.py to replay known-good
parameters first, then let Optuna continue exploring.

Usage:
    python seed_from_wandb.py                    # Seed top 5 trials
    python seed_from_wandb.py --top-n 3          # Seed top 3 only
    python seed_from_wandb.py --dry-run           # Preview without writing
    python seed_from_wandb.py --add-neighbors     # Also enqueue variations
"""

import argparse
import os
import sys

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

# ─── Configuration (must match hp_search.py) ──────────────────────────
VOLUME_DIR   = os.environ.get("HP_VOLUME_DIR", "/workspace")
DB_PATH      = os.environ.get("HP_DB_PATH", f"{VOLUME_DIR}/hp_search_results.db")
STORAGE_DB   = f"sqlite:///{DB_PATH}"
STUDY_NAME   = "pest-detection-hpsearch"
RANDOM_SEED  = 42

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "pest-detection-hpsearch")
WANDB_ENTITY  = os.environ.get("WANDB_ENTITY", "pfox1995-none")

# Valid values for categorical params (must match SEARCH_SPACE in hp_search.py)
VALID_CATEGORICALS = {
    "batch_size":       [1, 2, 4],
    "grad_accum":       [2, 4, 8],
    "num_epochs":       [2, 3, 5],
    "warmup_steps":     [0, 10, 50, 100],
    "lr_scheduler":     ["linear", "cosine", "cosine_with_restarts"],
    "max_seq_length":   [1024, 2048],
    "lora_r":           [8, 16, 32, 64],
    "lora_alpha_ratio": [1.0, 2.0, 4.0],
    "finetune_vision":  [True, False],
    "use_rslora":       [False, True],
}

# Continuous param ranges
CONTINUOUS_RANGES = {
    "learning_rate":    (5e-6, 5e-4),
    "weight_decay":     (0.0, 0.1),
    "crop_tight_prob":  (0.3, 0.7),
}


def snap_categorical(key, value):
    """Snap a value to the nearest valid categorical option."""
    if key not in VALID_CATEGORICALS:
        return value
    valid = VALID_CATEGORICALS[key]

    # Boolean snapping
    if isinstance(valid[0], bool):
        return bool(value)

    # Numeric snapping (find nearest)
    if isinstance(valid[0], (int, float)):
        return min(valid, key=lambda v: abs(v - value))

    # String — must match exactly
    if value in valid:
        return value
    # Fuzzy match for strings (e.g., underscores vs hyphens)
    for v in valid:
        if str(v).lower() == str(value).lower():
            return v
    return valid[0]  # fallback to first option


def clamp_continuous(key, value):
    """Clamp a continuous value to its valid range."""
    if key not in CONTINUOUS_RANGES:
        return value
    lo, hi = CONTINUOUS_RANGES[key]
    return max(lo, min(hi, value))


def extract_optuna_params(wandb_config):
    """Extract and validate Optuna-compatible params from a W&B config dict."""
    # Keys that Optuna's trial.suggest_* uses in hp_search.py
    OPTUNA_KEYS = [
        "batch_size", "grad_accum", "learning_rate", "num_epochs",
        "warmup_steps", "weight_decay", "lr_scheduler", "max_seq_length",
        "lora_r", "lora_alpha_ratio", "finetune_vision", "use_rslora",
        "crop_tight_prob",
    ]

    params = {}
    for key in OPTUNA_KEYS:
        if key not in wandb_config:
            return None  # Missing a required param
        value = wandb_config[key]

        if key in VALID_CATEGORICALS:
            value = snap_categorical(key, value)
        elif key in CONTINUOUS_RANGES:
            value = clamp_continuous(key, value)

        params[key] = value

    return params


def fetch_trials_from_wandb(top_n=5):
    """Fetch completed trial configs from W&B, sorted by eval_loss."""
    import wandb
    api = wandb.Api()

    entity_project = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
    print(f"  Querying W&B: {entity_project}")

    runs = api.runs(entity_project, filters={"state": "finished"})

    trials = []
    for r in runs:
        summary = r.summary._json_dict
        eval_loss = summary.get("final/eval_loss")
        accuracy  = summary.get("final/accuracy")

        if eval_loss is None or accuracy is None:
            continue

        params = extract_optuna_params(r.config)
        if params is None:
            print(f"  SKIP {r.name}: missing required params")
            continue

        trials.append({
            "name":      r.name,
            "eval_loss": eval_loss,
            "accuracy":  accuracy,
            "f1_macro":  summary.get("final/f1_macro", 0),
            "params":    params,
        })

    trials.sort(key=lambda t: t["eval_loss"])
    return trials[:top_n]


def generate_neighbors(best_params):
    """Generate parameter variations around the best config.

    Explores the immediate neighborhood: adjacent categorical values
    and +/- 50% on the learning rate. This helps Optuna map the
    performance landscape near the optimum faster.
    """
    neighbors = []

    # Variation 1: try adjacent lora_r values
    r_idx = VALID_CATEGORICALS["lora_r"].index(best_params["lora_r"])
    for offset in [-1, 1]:
        new_idx = r_idx + offset
        if 0 <= new_idx < len(VALID_CATEGORICALS["lora_r"]):
            n = best_params.copy()
            n["lora_r"] = VALID_CATEGORICALS["lora_r"][new_idx]
            neighbors.append(("lora_r_neighbor", n))

    # Variation 2: try each scheduler with the best params
    for sched in VALID_CATEGORICALS["lr_scheduler"]:
        if sched != best_params["lr_scheduler"]:
            n = best_params.copy()
            n["lr_scheduler"] = sched
            neighbors.append((f"scheduler_{sched}", n))

    # Variation 3: learning rate +/- 50%
    lr = best_params["learning_rate"]
    for factor, label in [(0.5, "lr_half"), (2.0, "lr_double")]:
        new_lr = clamp_continuous("learning_rate", lr * factor)
        n = best_params.copy()
        n["learning_rate"] = new_lr
        neighbors.append((label, n))

    # Variation 4: try with vision layers enabled/disabled (opposite of best)
    n = best_params.copy()
    n["finetune_vision"] = not best_params["finetune_vision"]
    neighbors.append(("toggle_vision", n))

    return neighbors


def seed_study(trials, neighbors=None):
    """Create/load Optuna study and enqueue trials."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_DB,
        direction="minimize",
        sampler=TPESampler(
            seed=RANDOM_SEED, n_startup_trials=5, multivariate=True,
        ),
        pruner=HyperbandPruner(
            min_resource=1, max_resource=5, reduction_factor=3,
        ),
        load_if_exists=True,
    )

    enqueued = 0
    for t in trials:
        study.enqueue_trial(t["params"])
        enqueued += 1
        print(f"  + {t['name']:15s} | eval_loss={t['eval_loss']:.6f} | acc={t['accuracy']:.4f}")

    if neighbors:
        print(f"\n  Neighbor variations (from best trial):")
        for label, params in neighbors:
            study.enqueue_trial(params)
            enqueued += 1
            print(f"  + {label}")

    return study, enqueued


def main():
    parser = argparse.ArgumentParser(
        description="Seed Optuna study with best W&B trials"
    )
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top trials to enqueue (default: 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be enqueued")
    parser.add_argument("--add-neighbors", action="store_true",
                        help="Also enqueue variations of the best params")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  W&B -> Optuna Seed")
    print(f"{'='*60}\n")

    # ─── Fetch from W&B ──────────────────────────────────────────────
    print(f"Fetching top {args.top_n} trials from W&B...")
    trials = fetch_trials_from_wandb(args.top_n)

    if not trials:
        print("\nNo completed trials found in W&B!")
        print("Check WANDB_API_KEY and project name.")
        sys.exit(1)

    print(f"\nFound {len(trials)} completed trials:\n")
    print(f"  {'#':>3} {'Name':15s} {'Eval Loss':>10} {'Accuracy':>9} {'F1':>7}")
    print(f"  {'-'*50}")
    for i, t in enumerate(trials):
        marker = " <-- BEST" if i == 0 else ""
        print(
            f"  {i+1:3d} {t['name']:15s} {t['eval_loss']:10.6f} "
            f"{t['accuracy']:9.4f} {t['f1_macro']:7.4f}{marker}"
        )

    # ─── Generate neighbors ──────────────────────────────────────────
    neighbors = None
    if args.add_neighbors and trials:
        neighbors = generate_neighbors(trials[0]["params"])
        print(f"\n  Will also enqueue {len(neighbors)} neighbor variations")

    # ─── Dry run ─────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n[DRY RUN] Would enqueue {len(trials)} trials", end="")
        if neighbors:
            print(f" + {len(neighbors)} neighbors", end="")
        print(". Run without --dry-run to proceed.")

        print(f"\nBest trial params for reference:")
        for k, v in sorted(trials[0]["params"].items()):
            print(f"  {k:25s} = {v}")
        return

    # ─── Seed the study ──────────────────────────────────────────────
    print(f"\nSeeding Optuna study '{STUDY_NAME}'...")
    print(f"  DB: {DB_PATH}\n")

    study, count = seed_study(trials, neighbors)

    print(f"\n{'='*60}")
    print(f"  Done! {count} trials enqueued.")
    print(f"  Optuna will run these FIRST, then explore new space.")
    print(f"")
    print(f"  Next: python hp_search.py --n-trials <N>")
    print(f"  (set N >= {count} to include all seeded + new trials)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
