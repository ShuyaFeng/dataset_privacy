"""
Phase 3 Task 3.3 — Compute DPRI features for all datasets.

Usage:
    python experiments/run_dpri.py
    python experiments/run_dpri.py --dataset adult   # single dataset
    python experiments/run_dpri.py --k 10            # change k-NN k

Output:
    results/dpri/dpri_features.csv   — one row per dataset
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dpri.features import compute_dpri_features

DATASETS = [
    "adult", "compas", "purchase100", "texas100",
    "nhanes", "movielens", "gowalla",
]


def load_dataset(name: str, data_dir: Path):
    path = data_dir / f"{name}.npz"
    if not path.exists():
        return None, None
    d = np.load(path)
    return d["X"], d["y"]


def merge_json_to_csv(out_dir: Path):
    """Combine all per-dataset *_dpri.json checkpoints into dpri_features.csv.

    Used after a Slurm array job where each task computed one dataset.
    """
    rows = []
    for jp in sorted(out_dir.glob("*_dpri.json")):
        with open(jp) as f:
            rows.append(json.load(f))
    if not rows:
        print(f"No *_dpri.json files found in {out_dir}")
        return
    df = pd.DataFrame(rows).set_index("dataset")
    csv_path = out_dir / "dpri_features.csv"
    df.to_csv(csv_path)
    print(f"Merged {len(rows)} datasets -> {csv_path}")
    print(df.round(4).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default=None, choices=DATASETS + [None])
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--out_dir",  default="results/dpri")
    parser.add_argument("--k",        type=int, default=5)
    parser.add_argument("--merge",    action="store_true",
                        help="merge existing *_dpri.json into dpri_features.csv and exit")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # merge mode: combine per-dataset JSON checkpoints into the final CSV
    if args.merge:
        merge_json_to_csv(out_dir)
        return

    targets = [args.dataset] if args.dataset else DATASETS
    rows = []

    for name in targets:
        print(f"\n{'='*50}")
        print(f"Dataset: {name}")
        X, y = load_dataset(name, data_dir)
        if X is None:
            print(f"  SKIP — {name}.npz not found in {data_dir}")
            continue

        feats = compute_dpri_features(X, y, k=args.k)
        feats["dataset"] = name
        feats["n_samples"] = len(X)
        feats["n_features"] = X.shape[1]
        rows.append(feats)

        # also save individual JSON for checkpointing
        with open(out_dir / f"{name}_dpri.json", "w") as f:
            json.dump(feats, f, indent=2)
        print(f"  Done. uniqueness_mean={feats['uniqueness_mean']:.4f}  "
              f"cluster_sep={feats['cluster_sep']:.4f}")

    # Only write the combined CSV when running ALL datasets in one process.
    # In Slurm array mode (one --dataset per task), run with --merge afterwards.
    if not args.dataset and rows:
        df = pd.DataFrame(rows).set_index("dataset")
        csv_path = out_dir / "dpri_features.csv"
        df.to_csv(csv_path)
        print(f"\nSaved: {csv_path}")
        print(df.round(4).to_string())


if __name__ == "__main__":
    main()
