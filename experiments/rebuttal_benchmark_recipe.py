"""
Rebuttal (S&P 2027 #1613): does the benchmark construction recipe raise
uniqueness and risk?  (Corollary 1, Reviewer C: "isn't really proved")

Purchase100 / Texas100 were built by (1) clustering records into 100
balanced classes and (2) keeping a subsample.  Corollary 1 claims each step
pushes the geometric factor up.  We apply exactly that recipe to ordinary
datasets and measure what happens to the DPRI features and to the measured
risk, which turns the corollary from an argument into an experiment.

Steps, on a dataset D with n >= 10k:
  original      : as in the corpus
  step1_cluster : relabel with k-means (C classes), keep all rows
  step2_balance : step1 + balance classes by subsampling to the smallest
                  class (capped at PER_CLASS rows per class)
  subsample_only: original labels, uniform random subsample to the same n
                  as step2 (control: is it subsampling or clustering?)

For each variant: uniqueness, density, cluster separation (k=5), and
loss-threshold AUC with MLP / XGBoost / RF.

Usage:
    python experiments/rebuttal_benchmark_recipe.py --dataset adult
Output:
    results/rebuttal/recipe/{dataset}.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dpri.features import (compute_uniqueness, compute_density,  # noqa: E402
                               compute_cluster_separation)
from src.mia.attacks import attack_loss_threshold, tpr_at_fpr  # noqa: E402
from src.models.classifiers import get_model                 # noqa: E402

OUT_DIR = Path("results/rebuttal/recipe")
CANDIDATES = ["adult", "covtype", "nomao", "bankmarketing", "letter",
              "electricity", "magic", "mnist", "gowalla"]
N_CLUSTERS = 100
PER_CLASS = 200          # 100 classes x 200 = 20k rows, Purchase100-like scale
MODELS = ["mlp", "xgboost", "rf"]


def geometry(X, y):
    u = compute_uniqueness(X, k=5)
    rho = compute_density(X, k=5)
    return {"uniqueness_mean": float(u.mean()), "density_mean": float(rho.mean()),
            "cluster_sep": compute_cluster_separation(X, y),
            "n_samples": int(len(X)), "n_classes": int(len(np.unique(y)))}


def risk(X, y, seed=42):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    out = {}
    for m in MODELS:
        t0 = time.time()
        model = get_model(m, n_classes=len(np.unique(y)), seed=seed).fit(Xtr, ytr)
        fpr, tpr, auc, det = attack_loss_threshold(model, Xtr, ytr, Xte, yte, return_details=True)
        out[m] = {"auc": float(auc), "tpr_at_fpr_01": tpr_at_fpr(fpr, tpr, 0.01),
                  "acc_gap": det["train_acc"] - det["test_acc"],
                  "elapsed_sec": round(time.time() - t0, 1)}
    out["mean_auc"] = float(np.mean([out[m]["auc"] for m in MODELS]))
    reg = [out[m]["auc"] for m in ("mlp", "xgboost") if m in out]
    out["mean_auc_regularized"] = float(np.mean(reg)) if reg else float("nan")
    return out


def balance(X, y, rng, per_class):
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        take = min(len(ci), per_class)
        idx.append(rng.choice(ci, take, replace=False))
    idx = np.concatenate(idx)
    return X[idx], y[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=CANDIDATES)
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_clusters", type=int, default=N_CLUSTERS)
    ap.add_argument("--per_class", type=int, default=PER_CLASS)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}.json"
    if out_path.exists():
        print(f"Already done: {out_path}")
        return

    d = np.load(Path(args.data_dir) / f"{args.dataset}.npz")
    X, y = d["X"].astype(np.float32), d["y"]
    rng = np.random.default_rng(args.seed)
    variants = {}

    print(f"[{args.dataset}] original n={len(X)}", flush=True)
    variants["original"] = (X, y)

    km = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=3)
    y_c = km.fit_predict(X)
    # drop clusters with < 2 members so stratified splits work
    keep = np.isin(y_c, np.where(np.bincount(y_c) >= 4)[0])
    variants["step1_cluster"] = (X[keep], y_c[keep])

    Xb, yb = balance(X[keep], y_c[keep], rng, args.per_class)
    variants["step2_cluster_balance"] = (Xb, yb)

    sub = rng.choice(len(X), len(Xb), replace=False)
    variants["subsample_only_control"] = (X[sub], y[sub])

    out = {"dataset": args.dataset, "seed": args.seed, "n_clusters": args.n_clusters,
           "per_class": args.per_class, "variants": {}}
    for name, (Xv, yv) in variants.items():
        t0 = time.time()
        g = geometry(Xv, yv)
        r = risk(Xv, yv, args.seed)
        out["variants"][name] = {"geometry": g, "risk": r, "elapsed_sec": round(time.time() - t0, 1)}
        print(f"  {name:<24} n={g['n_samples']:<6} u={g['uniqueness_mean']:.3f} "
              f"rho={g['density_mean']:.3f} S={g['cluster_sep']:+.3f} "
              f"meanAUC={r['mean_auc']:.3f} regAUC={r['mean_auc_regularized']:.3f}", flush=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
