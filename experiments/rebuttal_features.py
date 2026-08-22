"""
Rebuttal (#1613): per-dataset geometric quantities for rebuttal_experiments.py.

One process per dataset (Slurm array, slurm/rebuttal_features_array.sh):
    python experiments/rebuttal_features.py --dataset adult
    -> results/rebuttal/features/adult.json

Everything is computed from the raw standardized matrix in data/processed/<name>.npz
(the same input the DPRI features and the attack grid use). One k-NN search
with k = KMAX is reused for all k-dependent quantities.

Quantities (and the rebuttal item they serve):
  k_sweep[k]      mean uniqueness u_k, mean floored density rho_k (paper surrogate),
                  mean k-NN label purity                           (Reviewer D: k; Admin 1)
  floor           density-floor diagnostics at k=5                 (Reviewer D, Q2)
  entropy_bins    feature entropy for several bin counts           (Reviewer D)
  pca             u, rho, purity at k=5 after PCA to <=50 dims     (Reviewer D: pixel space)
  formula         theorem factor g(x)=u/rho^{1/d}, surrogate and volume-based density,
                  dataset means of g and log g                     (Admin item 3)
  split           in-sample vs out-of-sample 5-NN distance on the attack grid's
                  50/50 stratified split (seed 42)                 (Reviewer C: two distances)
"""

import argparse
import json
import sys
import time
from math import lgamma, log, pi
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dpri.features import compute_entropy  # noqa: E402

K_LIST = [3, 5, 10, 20]
KMAX = max(K_LIST)
BIN_LIST = [10, 20, 50, 100, 200]
PCA_DIMS = 50
K_PAPER = 5


def knn_no_self(X: np.ndarray, k: int):
    """Distances and indices of the k nearest neighbours, self excluded.
    Column j holds the (j+1)-th neighbour, matching features.compute_uniqueness
    (which reads dists[:, k] with self in column 0)."""
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1).fit(X)
    dists, idx = nn.kneighbors(X)
    return dists[:, 1:], idx[:, 1:]


def floored_density(r_k: np.ndarray):
    """Paper's density surrogate 1/r_k with the 10%-of-median floor
    (verbatim logic of src/dpri/features.compute_density)."""
    nonzero = r_k[r_k > 1e-12]
    floor = float(np.median(nonzero)) * 0.1 if nonzero.size > 0 else 1e-6
    r_f = np.maximum(r_k, floor)
    return 1.0 / r_f, r_f, floor


def purity(idx: np.ndarray, y: np.ndarray, k: int) -> float:
    """Mean over samples of the fraction of the k nearest neighbours sharing the label."""
    same = (y[idx[:, :k]] == y[:, None])
    return float(same.mean())


def k_sweep(dists, idx, y):
    out = {}
    for k in K_LIST:
        r_k = dists[:, k - 1]
        rho, _, _ = floored_density(r_k)
        out[str(k)] = {
            "u_mean": float(r_k.mean()),
            "u_p90": float(np.percentile(r_k, 90)),
            "rho_mean": float(rho.mean()),
            "purity_mean": purity(idx, y, k),
        }
    return out


def floor_diagnostics(dists):
    r_k = dists[:, K_PAPER - 1]
    rho, r_f, floor = floored_density(r_k)
    nonzero = r_k[r_k > 1e-12]
    capped = 1.0 / np.maximum(r_k, 1e-6)       # no floor, only a numerical cap
    return {
        "k": K_PAPER,
        "median_r_k": float(np.median(nonzero)) if nonzero.size else 0.0,
        "floor_value": floor,
        "frac_at_floor": float(np.mean(r_k < floor)),
        "frac_exact_duplicate": float(np.mean(dists[:, 0] <= 1e-9)),
        "density_mean_floored": float(rho.mean()),
        "density_mean_capped_1e-6": float(capped.mean()),
        "density_median_floored": float(np.median(rho)),
    }


def entropy_bins(X):
    return {str(b): compute_entropy(X, n_bins=b) for b in BIN_LIST}


def pca_block(X, y):
    n, d = X.shape
    ncomp = int(min(PCA_DIMS, d, n - 1))
    Xp = PCA(n_components=ncomp, random_state=42).fit_transform(X)
    dists, idx = knn_no_self(Xp, K_PAPER)
    r_k = dists[:, K_PAPER - 1]
    rho, _, _ = floored_density(r_k)
    return {
        "n_components": ncomp,
        "u_mean": float(r_k.mean()),
        "rho_mean": float(rho.mean()),
        "purity_mean": purity(idx, y, K_PAPER),
    }


def theorem_formula(dists, n, d):
    """g(x) = u(x) / rho(x)^{1/d} at k=5.
    surrogate: rho = 1/r_f  ->  g = r * r_f^{1/d}
    volume:    rho = k / (n V_d r^d) -> rho^{1/d} = (k/(n V_d))^{1/d} / r
               -> g = r^2 (n V_d / k)^{1/d}, computed in log space."""
    k = K_PAPER
    r = dists[:, k - 1]
    _, r_f, _ = floored_density(r)
    r_safe = np.maximum(r, 1e-12)
    log_Vd = (d / 2.0) * log(pi) - lgamma(d / 2.0 + 1.0)
    log_g_sur = np.log(r_safe) + np.log(r_f) / d
    log_g_vol = 2.0 * np.log(r_safe) + (log(n) + log_Vd - log(k)) / d
    return {
        "k": k,
        "log_Vd": log_Vd,
        "g_surrogate_mean": float(np.exp(log_g_sur).mean()),
        "log_g_surrogate_mean": float(log_g_sur.mean()),
        "g_volume_mean": float(np.exp(log_g_vol).mean()),
        "log_g_volume_mean": float(log_g_vol.mean()),
    }


def split_distances(X, y, seed=42):
    """Replicates run_mia_grid's split exactly, then compares the member->member
    5-NN distance (u_in, the paper's u) with the member->non-member 5-NN distance
    (u_out, the quantity Theorem 1's Step 2 actually needs)."""
    Xm, Xn, _, _ = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    k = K_PAPER
    d_in, _ = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(Xm).kneighbors(Xm)
    u_in = d_in[:, k]                       # self in column 0
    d_out, _ = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(Xn).kneighbors(Xm)
    u_out = d_out[:, k - 1]
    ratio = float(u_in.mean() / u_out.mean()) if u_out.mean() > 0 else float("nan")
    return {
        "k": k,
        "n_members": int(len(Xm)),
        "n_nonmembers": int(len(Xn)),
        "u_in_mean": float(u_in.mean()),
        "u_out_mean": float(u_out.mean()),
        "ratio_in_over_out": ratio,
        "u_in_median": float(np.median(u_in)),
        "u_out_median": float(np.median(u_out)),
        "per_sample_spearman": float(_spearman(u_in, u_out)),
    }


def _spearman(a, b):
    from scipy.stats import spearmanr
    return spearmanr(a, b)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default="results/rebuttal/features")
    ap.add_argument("--skip_split", action="store_true")
    ap.add_argument("--skip_pca", action="store_true")
    args = ap.parse_args()

    path = Path(args.data_dir) / f"{args.dataset}.npz"
    if not path.exists():
        print(f"SKIP: {path} not found")
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}.json"

    d = np.load(path)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    n, dim = X.shape
    print(f"{args.dataset}: n={n} d={dim} classes={len(np.unique(y))}")

    t0 = time.time()
    dists, idx = knn_no_self(X, KMAX)
    print(f"  kNN(k={KMAX}) done in {time.time()-t0:.1f}s")

    res = {
        "dataset": args.dataset,
        "n_samples": int(n),
        "n_features": int(dim),
        "n_classes": int(len(np.unique(y))),
        "k_sweep": k_sweep(dists, idx, y),
        "floor": floor_diagnostics(dists),
        "entropy_bins": entropy_bins(X),
        "formula": theorem_formula(dists, n, dim),
    }
    if not args.skip_pca:
        t0 = time.time()
        res["pca"] = pca_block(X, y)
        print(f"  PCA block done in {time.time()-t0:.1f}s")
    if not args.skip_split:
        t0 = time.time()
        try:
            res["split"] = split_distances(X, y)
        except ValueError as e:          # stratify impossible (class with 1 sample)
            res["split"] = {"error": str(e)}
        print(f"  split block done in {time.time()-t0:.1f}s")

    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
