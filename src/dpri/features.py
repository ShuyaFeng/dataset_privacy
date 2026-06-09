"""
DPRI Feature Computation (Phase 3, Task 3.3)

Five dataset-level features, each aggregated to (mean, p90):
  1. Sample Uniqueness  u(x)  — k-NN nearest-neighbour distance
  2. Local Density      ρ(x)  — k-NN density estimate
  3. Outlier Score      o(x)  — ensemble of LOF + IsolationForest
  4. Feature Entropy    H(X)  — per-feature Shannon entropy, averaged
  5. Cluster Separation S     — silhouette score on class labels

All features are computed on raw (preprocessed) X, y without any model.
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score


# ── 1. Sample Uniqueness ────────────────────────────────────────────────────

def compute_uniqueness(X: np.ndarray, k: int = 5) -> np.ndarray:
    """
    For each sample, return the distance to its k-th nearest neighbour.
    High value → sample is isolated → higher privacy risk.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1)
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    # dists[:, 0] is self (distance=0), dists[:, k] is k-th neighbour
    return dists[:, k]


# ── 2. Local Density ────────────────────────────────────────────────────────

def compute_density(X: np.ndarray, k: int = 5) -> np.ndarray:
    """
    k-NN density estimate: ρ(x) = k / (n * Vol(B(x, r_k(x)))).
    Simplified to 1 / r_k(x) (monotonically equivalent, avoids d-dependent ball volume).
    Low value → sparse region → higher privacy risk.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1)
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    r_k = dists[:, k]
    # Floor r_k at a small fraction of the typical scale. Duplicate or
    # near-duplicate points have r_k ~ 0, and a naive 1/r_k blows up the
    # density (e.g. millions on COMPAS/Gowalla, which have many repeated rows).
    # Flooring at 10% of the median keeps duplicates "very dense" but bounded.
    nonzero = r_k[r_k > 1e-12]
    floor = float(np.median(nonzero)) * 0.1 if nonzero.size > 0 else 1e-6
    r_k = np.maximum(r_k, floor)
    return 1.0 / r_k


# ── 3. Outlier Score ────────────────────────────────────────────────────────

def compute_outlier_score(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Ensemble of LOF and IsolationForest.
    Both return scores in [-1, 1] (more negative = more outlier).
    We flip sign so high value = more outlier-like = higher risk.
    Returned score is in [0, 1] after min-max normalization.
    """
    n = len(X)
    n_neighbors_lof = min(20, n - 1)

    # LOF: negative_outlier_factor_ is <= -1; more negative = more outlier
    lof = LocalOutlierFactor(n_neighbors=n_neighbors_lof, novelty=False, n_jobs=-1)
    lof.fit(X)
    lof_scores = -lof.negative_outlier_factor_  # flip: high = outlier

    # IsolationForest: score_samples returns anomaly score (lower = more anomalous)
    iso = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
    iso.fit(X)
    iso_scores = -iso.score_samples(X)  # flip: high = anomalous

    # Normalize each to [0, 1] then average
    def _minmax(s):
        lo, hi = s.min(), s.max()
        if hi - lo < 1e-12:
            return np.zeros_like(s)
        return (s - lo) / (hi - lo)

    return (_minmax(lof_scores) + _minmax(iso_scores)) / 2.0


# ── 4. Feature Entropy ──────────────────────────────────────────────────────

def compute_entropy(X: np.ndarray, n_bins: int = 50) -> float:
    """
    Average Shannon entropy across all features.
    Each continuous feature is discretized into n_bins bins.
    High entropy → more uncertainty → generally higher identifiability.
    Returns a single scalar (dataset-level, not per-sample).
    """
    entropies = []
    for j in range(X.shape[1]):
        col = X[:, j]
        counts, _ = np.histogram(col, bins=n_bins)
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        entropies.append(scipy_entropy(probs, base=2))
    return float(np.mean(entropies))


# ── 5. Cluster Separation ───────────────────────────────────────────────────

def compute_cluster_separation(X: np.ndarray, y: np.ndarray) -> float:
    """
    Silhouette score using class labels as clusters.
    Range [-1, 1]; high value = well-separated classes = higher boundary risk.
    Returns 0.0 if only one class present (degenerate).
    """
    n_classes = len(np.unique(y))
    if n_classes < 2:
        return 0.0
    # silhouette_score is O(n^2); subsample if large
    max_n = 5000
    if len(X) > max_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), max_n, replace=False)
        X_s, y_s = X[idx], y[idx]
    else:
        X_s, y_s = X, y
    return float(silhouette_score(X_s, y_s))


# ── Aggregate to dataset-level DPRI feature vector ─────────────────────────

def compute_dpri_features(X: np.ndarray, y: np.ndarray, k: int = 5) -> dict:
    """
    Compute all DPRI features and aggregate to dataset level.

    Returns a dict with keys:
      uniqueness_mean, uniqueness_p90,
      density_mean,    density_p90,
      outlier_mean,    outlier_p90,
      entropy,                         (scalar)
      cluster_sep,                     (scalar)

    This dict is the feature vector fed to the regression in Task 3.4.
    """
    print("  [DPRI] Computing uniqueness ...")
    u = compute_uniqueness(X, k=k)

    print("  [DPRI] Computing density ...")
    rho = compute_density(X, k=k)

    print("  [DPRI] Computing outlier scores ...")
    o = compute_outlier_score(X)

    print("  [DPRI] Computing entropy ...")
    h = compute_entropy(X)

    print("  [DPRI] Computing cluster separation ...")
    s = compute_cluster_separation(X, y)

    return {
        "uniqueness_mean": float(np.mean(u)),
        "uniqueness_p90":  float(np.percentile(u, 90)),
        "density_mean":    float(np.mean(rho)),
        "density_p90":     float(np.percentile(rho, 90)),
        "outlier_mean":    float(np.mean(o)),
        "outlier_p90":     float(np.percentile(o, 90)),
        "entropy":         h,
        "cluster_sep":     s,
    }
