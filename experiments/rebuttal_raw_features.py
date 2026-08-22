"""
Rebuttal (S&P 2027 Cycle 1, #1613): per-dataset geometric variants.

One Slurm task per dataset. Everything here is computed from the raw
(preprocessed) feature matrix, with NO model training, so it stays inside
the pre-training contract of DPRI. Output feeds
experiments/rebuttal_experiments.py, which aggregates across datasets.

What is computed (and which review point it answers):

  k_variants        u_k, rho_k for k in {3,5,10,20}        Reviewer D (sensitivity)
  entropy_bins      H for bins in {10,20,50,100,200}       Reviewer D (sensitivity)
  aggregation       median / p90 / trimmed-mean of u, rho  Reviewer B (feature choice)
  formula_exact     mean_x u(x) * rho(x)^(-1/d)            Reviewer C / Admin 3
  pca               u, rho, S on PCA-50 embedding          Reviewer D (pixel distance)
  normalized        u / median pairwise dist, u / sqrt(d)  Reviewer D (high-d artefact)
  distances         u_in (k-NN among members) vs           Reviewer C (proof conflates
                    u_out (k-NN among non-members) under     two distances)
                    the exact seed-42 50/50 split
  duplicates        exact-duplicate fraction, floored       Reviewer D (Gowalla vs
                    fraction, density under other floors     Movielens density)
  label_proxy       k-NN label disagreement, 1-NN error     Admin 1 (failure analysis:
                                                             easily learnable tasks)

Usage:
    python experiments/rebuttal_raw_features.py --dataset adult
    python experiments/rebuttal_raw_features.py --all          # small datasets only
Output:
    results/rebuttal/raw/{dataset}_raw.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dpri.features import compute_entropy  # noqa: E402

DATASETS = [
    "adult", "compas", "purchase100", "texas100",
    "nhanes", "movielens", "gowalla",
    "covtype", "digits", "creditg", "spambase", "mushroom", "electricity",
    "letter", "optdigits", "pendigits", "satimage", "segment", "vehicle",
    "ionosphere", "phoneme", "bankmarketing", "magic", "nomao", "har",
    "gasdrift", "mnist", "fashionmnist", "jm1", "kc1", "breastw",
]

K_VALUES = [3, 5, 10, 20]
BIN_VALUES = [10, 20, 50, 100, 200]
FLOOR_FRACS = [0.0, 0.05, 0.1, 0.2, 0.5]
PCA_DIM = 50
SPLIT_SEED = 42          # must match run_mia_grid.py
OUT_DIR = Path("results/rebuttal/raw")


# ── helpers ─────────────────────────────────────────────────────────────────

def _knn_dists(X, k_max, n_jobs=-1):
    """Distances to the 1..k_max nearest neighbours (self excluded), and
    the neighbour indices."""
    nn = NearestNeighbors(n_neighbors=k_max + 1, algorithm="auto", n_jobs=n_jobs)
    nn.fit(X)
    d, idx = nn.kneighbors(X)
    return d[:, 1:], idx[:, 1:]


def _density_from_rk(r_k, floor_frac):
    """Paper's density surrogate 1/r_k with r_k floored at floor_frac * median
    of the non-zero r_k (floor_frac=0.1 is the paper's choice)."""
    nonzero = r_k[r_k > 1e-12]
    if floor_frac <= 0:
        floor = 1e-12
    else:
        floor = float(np.median(nonzero)) * floor_frac if nonzero.size else 1e-6
    return 1.0 / np.maximum(r_k, floor), float(floor)


def _silhouette(X, y, seed=42, max_n=5000):
    if len(np.unique(y)) < 2:
        return 0.0
    if len(X) > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), max_n, replace=False)
        X, y = X[idx], y[idx]
    return float(silhouette_score(X, y))


def _median_pairwise(X, seed=0, m=2000):
    rng = np.random.default_rng(seed)
    if len(X) > m:
        X = X[rng.choice(len(X), m, replace=False)]
    # pairwise distances on the subsample (m^2 / 2 entries)
    sq = (X ** 2).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
    np.maximum(D2, 0, out=D2)
    iu = np.triu_indices(len(X), k=1)
    return float(np.median(np.sqrt(D2[iu])))


def _dup_fraction(X):
    hashes = np.fromiter((hash(row.tobytes()) for row in np.ascontiguousarray(X)),
                         dtype=np.int64, count=len(X))
    return 1.0 - len(np.unique(hashes)) / len(X)


def _trimmed_mean(a, frac=0.1):
    a = np.sort(a)
    cut = int(len(a) * frac)
    return float(a[cut:len(a) - cut].mean()) if len(a) - 2 * cut > 0 else float(a.mean())


# ── main computation ────────────────────────────────────────────────────────

def compute_all(X, y, n_jobs=-1, verbose=True):
    n, d = X.shape
    out = {"n_samples": int(n), "n_features": int(d),
           "n_classes": int(len(np.unique(y)))}
    t0 = time.time()

    # 1. one k-NN query with k_max = 20 covers all k variants and the label proxy
    k_max = max(K_VALUES)
    dists, nbrs = _knn_dists(X, k_max, n_jobs)
    if verbose:
        print(f"  knn (k={k_max}) done in {time.time()-t0:.1f}s", flush=True)

    kv = {}
    for k in K_VALUES:
        r_k = dists[:, k - 1]
        rho_k, floor = _density_from_rk(r_k, 0.1)
        kv[str(k)] = {
            "uniqueness_mean": float(r_k.mean()),
            "density_mean": float(rho_k.mean()),
            "uniqueness_median": float(np.median(r_k)),
            "density_median": float(np.median(rho_k)),
            "uniqueness_p90": float(np.percentile(r_k, 90)),
            "density_p90": float(np.percentile(rho_k, 90)),
            "uniqueness_trimmed": _trimmed_mean(r_k),
            "density_trimmed": _trimmed_mean(rho_k),
            "floor": floor,
            # Theorem 1's per-sample geometric factor u(x) / rho(x)^{1/d},
            # averaged over samples (Eq. risk_bound). With the 1/r_k
            # surrogate this is mean r_k^{1 + 1/d} up to the floor.
            "formula_exact_mean": float(np.mean(r_k * np.power(rho_k, -1.0 / d))),
            "formula_exact_median": float(np.median(r_k * np.power(rho_k, -1.0 / d))),
        }
    out["k_variants"] = kv

    # 2. entropy bins
    out["entropy_bins"] = {str(b): compute_entropy(X, n_bins=b) for b in BIN_VALUES}

    # 3. duplicates and floors (k = 5, the paper's setting)
    r5 = dists[:, 4]
    # brute-force kNN in high d returns ~1e-3 numerical noise for identical
    # rows, so "zero" is relative to the dataset's scale
    med_nz = float(np.median(r5[r5 > 1e-12])) if (r5 > 1e-12).any() else 1.0
    dup = {
        "exact_duplicate_fraction": _dup_fraction(X),
        "frac_r5_zero": float((r5 <= 1e-4 * med_nz).mean()),
        "median_r5_nonzero": float(np.median(r5[r5 > 1e-12])) if (r5 > 1e-12).any() else 0.0,
        "density_by_floor": {},
    }
    for f in FLOOR_FRACS:
        rho_f, floor = _density_from_rk(r5, f)
        dup["density_by_floor"][str(f)] = {
            "mean": float(np.mean(np.minimum(rho_f, 1e12))),
            "median": float(np.median(rho_f)),
            "frac_at_floor": float((r5 <= floor + 1e-15).mean()),
        }
    out["duplicates"] = dup

    # 4. label proxy (pre-training "task difficulty" from the same k-NN index)
    yn = y[nbrs]                                   # (n, k_max) neighbour labels
    out["label_proxy"] = {
        "knn5_label_disagreement": float((yn[:, :5] != y[:, None]).mean()),
        "knn20_label_disagreement": float((yn[:, :20] != y[:, None]).mean()),
        "nn1_error": float((yn[:, 0] != y).mean()),
        "frac_samples_all5_disagree": float((yn[:, :5] != y[:, None]).all(1).mean()),
    }

    # 5. normalized uniqueness (dimension-free scale)
    med_pd = _median_pairwise(X)
    out["normalized"] = {
        "median_pairwise_distance": med_pd,
        "uniqueness_over_median_pairwise": float(r5.mean() / med_pd) if med_pd > 0 else None,
        "uniqueness_over_sqrt_d": float(r5.mean() / np.sqrt(d)),
    }

    # 6. u_in vs u_out under the exact MIA split
    idx_all = np.arange(n)
    mem_idx, non_idx = train_test_split(idx_all, test_size=0.5,
                                        random_state=SPLIT_SEED, stratify=y)
    Xm, Xn = X[mem_idx], X[non_idx]
    k = 5
    nn_in = NearestNeighbors(n_neighbors=k + 1, n_jobs=n_jobs).fit(Xm)
    d_in = nn_in.kneighbors(Xm)[0][:, k]           # k-th other member
    nn_out = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs).fit(Xn)
    d_out = nn_out.kneighbors(Xm)[0][:, k - 1]     # k-th non-member
    # nearest-neighbour (k=1) version too, which is what the proof's u(x) is
    d_in1 = nn_in.kneighbors(Xm, n_neighbors=2)[0][:, 1]
    d_out1 = nn_out.kneighbors(Xm, n_neighbors=1)[0][:, 0]
    sp5 = spearmanr(d_in, d_out)[0] if n > 10 else float("nan")
    sp1 = spearmanr(d_in1, d_out1)[0] if n > 10 else float("nan")
    out["distances"] = {
        "k": k,
        "n_members": int(len(mem_idx)),
        "u_in_mean_k5": float(d_in.mean()),
        "u_out_mean_k5": float(d_out.mean()),
        "ratio_mean_k5": float(d_in.mean() / d_out.mean()) if d_out.mean() > 0 else None,
        "per_sample_spearman_k5": float(sp5),
        "u_in_mean_k1": float(d_in1.mean()),
        "u_out_mean_k1": float(d_out1.mean()),
        "ratio_mean_k1": float(d_in1.mean() / d_out1.mean()) if d_out1.mean() > 0 else None,
        "per_sample_spearman_k1": float(sp1),
        # full-data uniqueness (what DPRI actually uses) vs the two split versions
        "u_full_mean_k5": float(r5.mean()),
    }
    if verbose:
        print(f"  distances done in {time.time()-t0:.1f}s", flush=True)

    # 7. PCA embedding variants (only meaningful when d > PCA_DIM)
    pca_dim = min(PCA_DIM, d, n - 1)
    if d > PCA_DIM:
        pca = PCA(n_components=pca_dim, random_state=0, svd_solver="randomized")
        Z = pca.fit_transform(X)
        dz, nz = _knn_dists(Z, 5, n_jobs)
        rz = dz[:, 4]
        rho_z, _ = _density_from_rk(rz, 0.1)
        out["pca"] = {
            "dim": int(pca_dim),
            "explained_variance": float(pca.explained_variance_ratio_.sum()),
            "uniqueness_mean": float(rz.mean()),
            "density_mean": float(rho_z.mean()),
            "cluster_sep": _silhouette(Z, y),
            "knn5_label_disagreement": float((y[nz] != y[:, None]).mean()),
            "median_pairwise_distance": _median_pairwise(Z),
        }
        out["pca"]["uniqueness_over_median_pairwise"] = (
            out["pca"]["uniqueness_mean"] / out["pca"]["median_pairwise_distance"]
            if out["pca"]["median_pairwise_distance"] > 0 else None)
    else:
        out["pca"] = {"dim": int(d), "note": "d <= PCA_DIM, identical to raw features",
                      "uniqueness_mean": float(r5.mean()),
                      "density_mean": float(_density_from_rk(r5, 0.1)[0].mean()),
                      "cluster_sep": _silhouette(X, y),
                      "knn5_label_disagreement": out["label_proxy"]["knn5_label_disagreement"],
                      "median_pairwise_distance": med_pd,
                      "uniqueness_over_median_pairwise": out["normalized"]["uniqueness_over_median_pairwise"]}
    if verbose:
        print(f"  pca done in {time.time()-t0:.1f}s", flush=True)

    out["elapsed_sec"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, choices=DATASETS + [None])
    ap.add_argument("--all", action="store_true",
                    help="run every dataset found in --data_dir in one process")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = DATASETS if args.all else ([args.dataset] if args.dataset else [])
    if not targets:
        ap.error("give --dataset NAME or --all")

    for name in targets:
        path = Path(args.data_dir) / f"{name}.npz"
        out_path = out_dir / f"{name}_raw.json"
        if out_path.exists() and not args.force:
            print(f"Already done: {out_path.name}")
            continue
        if not path.exists():
            print(f"SKIP {name}: {path} not found")
            continue
        d = np.load(path)
        X, y = d["X"].astype(np.float32), d["y"]
        print(f"\n=== {name}: X={X.shape}, classes={len(np.unique(y))}", flush=True)
        res = compute_all(X, y, n_jobs=args.n_jobs)
        res["dataset"] = name
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  saved {out_path}  ({res['elapsed_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
