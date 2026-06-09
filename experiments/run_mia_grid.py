"""
Phase 1 Task 1.1 — MIA Grid Experiment
=======================================
Runs 3 attacks × 3 models × N datasets and saves AUC scores.
This output becomes the ground truth Risk(D) for DPRI regression.

Usage (local, single dataset for testing):
    python experiments/run_mia_grid.py --dataset adult --attack loss_threshold --model mlp

Usage (full grid, meant to be called by Slurm array job):
    python experiments/run_mia_grid.py --dataset adult --attack lira --model xgboost

Results are saved to:
    results/mia_grid/{dataset}__{attack}__{model}.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.classifiers import get_model, MODEL_NAMES
from src.mia.attacks import (
    attack_loss_threshold,
    attack_shadow_model,
    attack_lira,
    ATTACK_NAMES,
)

DATASET_NAMES = [
    "adult", "compas", "purchase100", "texas100",
    "nhanes", "movielens", "gowalla",
    "covtype", "digits", "creditg", "spambase", "mushroom", "electricity",
    "letter", "optdigits", "pendigits", "satimage", "segment", "vehicle",
    "ionosphere", "phoneme", "bankmarketing", "magic", "nomao", "har",
    "gasdrift", "mnist", "fashionmnist", "jm1", "kc1", "breastw",
]


def load_dataset(name: str, data_dir: Path):
    path = data_dir / f"{name}.npz"
    if not path.exists():
        return None, None   # missing (e.g. an OpenML fetch that failed); caller skips
    d = np.load(path)
    return d["X"], d["y"]


def run_single(dataset: str, attack: str, model_name: str, data_dir: Path, seed: int = 42):
    X, y = load_dataset(dataset, data_dir)
    if X is None:
        return None

    # 50/50 member/non-member split — standard MIA convention
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y
    )

    n_classes = len(np.unique(y))
    t0 = time.time()

    if attack == "loss_threshold":
        model = get_model(model_name, n_classes=n_classes, seed=seed)
        model.fit(X_train, y_train)
        fpr, tpr, auc = attack_loss_threshold(model, X_train, y_train, X_test, y_test)

    elif attack == "shadow_model":
        fpr, tpr, auc = attack_shadow_model(model_name, X, y, n_shadow=4, seed=seed)

    elif attack == "lira":
        fpr, tpr, auc = attack_lira(model_name, X, y, n_shadow=16, seed=seed)

    else:
        raise ValueError(f"Unknown attack: {attack}")

    elapsed = time.time() - t0

    result = {
        "dataset": dataset,
        "attack": attack,
        "model": model_name,
        "auc": auc,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_classes": n_classes,
        "seed": seed,
        "elapsed_sec": round(elapsed, 2),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  required=True, choices=DATASET_NAMES)
    parser.add_argument("--attack",   required=True, choices=ATTACK_NAMES)
    parser.add_argument("--model",    required=True, choices=MODEL_NAMES)
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--out_dir",  default="results/mia_grid")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}__{args.attack}__{args.model}.json"

    if out_path.exists():
        print(f"Already done: {out_path.name} — skipping.")
        return

    print(f"Running: dataset={args.dataset} attack={args.attack} model={args.model}")
    result = run_single(
        args.dataset, args.attack, args.model,
        Path(args.data_dir), args.seed
    )
    if result is None:
        print(f"  SKIP: {args.dataset} not found (download may have failed) — exiting cleanly")
        return
    print(f"  AUC={result['auc']:.4f}  time={result['elapsed_sec']}s")

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
