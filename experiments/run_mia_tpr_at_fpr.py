"""
Rebuttal (#1613), Reviewer B: high-resolution LiRA pass for TPR at low FPR.

The main grid (run_mia_grid.py) already records TPR@{10,1,0.1}%FPR for every
configuration with LiRA's submission setting of 500 scored target samples.
At 0.1% FPR that resolution rests on a handful of negatives, so this script
re-runs LiRA only, with 2,000 scored target samples (16 shadow models as before),
for every dataset x model, and writes to a separate directory.

Single config (Slurm array, slurm/rebuttal_tpr_array.sh):
    python experiments/run_mia_tpr_at_fpr.py --dataset adult --model mlp
Aggregate (also consumed by rebuttal_experiments.py):
    python experiments/run_mia_tpr_at_fpr.py --aggregate
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.classifiers import MODEL_NAMES  # noqa: E402
from src.mia.attacks import attack_lira, tpr_at_fpr  # noqa: E402

OUT_DIR = Path("results/rebuttal/tpr_at_fpr")


def run_single(dataset, model_name, data_dir, seed=42, eval_n=2000, n_shadow=16):
    path = data_dir / f"{dataset}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    X, y = d["X"], d["y"]
    t0 = time.time()
    fpr, tpr, auc = attack_lira(model_name, X, y, n_shadow=n_shadow, seed=seed, eval_n=eval_n)
    return {
        "dataset": dataset, "model": model_name, "attack": "lira",
        "auc": float(auc),
        "tpr_at_fpr_10": tpr_at_fpr(fpr, tpr, 0.10),
        "tpr_at_fpr_01": tpr_at_fpr(fpr, tpr, 0.01),
        "tpr_at_fpr_001": tpr_at_fpr(fpr, tpr, 0.001),
        "lira_eval_n": int(min(eval_n, len(X))), "n_shadow": n_shadow,
        "elapsed_sec": round(time.time() - t0, 2),
    }


def aggregate():
    recs = []
    for p in OUT_DIR.glob("*.json"):
        if p.name == "tpr_fpr_summary.json":
            continue
        recs.append(json.load(open(p)))
    if not recs:
        print("No results found.")
        return
    df = pd.DataFrame(recs)
    print(f"Loaded {len(df)} results across {df['dataset'].nunique()} datasets")
    summary = {m: df.groupby("dataset")[m].mean().round(4).to_dict()
               for m in ["auc", "tpr_at_fpr_10", "tpr_at_fpr_01", "tpr_at_fpr_001"]}
    with open(OUT_DIR / "tpr_fpr_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(df.groupby("model")[["auc", "tpr_at_fpr_01", "tpr_at_fpr_001"]].mean().round(4).to_string())
    print(f"Saved: {OUT_DIR / 'tpr_fpr_summary.json'}  (DPRI correlation: experiments/rebuttal_experiments.py --only robustness)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--model", default=None, choices=MODEL_NAMES + [None])
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_n", type=int, default=2000)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate()
        return
    if not args.dataset or not args.model:
        sys.exit("ERROR: --dataset and --model required (or use --aggregate)")
    out_path = OUT_DIR / f"{args.dataset}__lira__{args.model}.json"
    if out_path.exists():
        print(f"Already done: {out_path.name}")
        return
    print(f"Running high-res LiRA: dataset={args.dataset} model={args.model} eval_n={args.eval_n}")
    res = run_single(args.dataset, args.model, Path(args.data_dir), args.seed, args.eval_n)
    if res is None:
        print(f"  SKIP: {args.dataset} not found")
        return
    print(f"  AUC={res['auc']:.4f}  TPR@1%FPR={res['tpr_at_fpr_01']:.4f}  TPR@0.1%FPR={res['tpr_at_fpr_001']:.4f}")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
