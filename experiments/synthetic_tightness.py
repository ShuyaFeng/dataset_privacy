"""
Synthetic tightness experiment for Proposition 2 (Task 2.2).

Proposition 2 claims: for data uniform on a d-dimensional hypercube, the
bound's geometric factor and the measured MIA advantage both scale as
n^{-1/d}.  This script tests that claim directly.

Setup:
  - X ~ U[0,1]^d, binary labels from a fixed random hyperplane + 10% label
    noise (the noise gives the model something to memorize, per Feldman).
  - Model: sklearn MLPClassifier (64-64), trained to convergence, no early
    stopping — the standard regularized model family of the main grid.
  - Measured advantage: loss-threshold attack (Yeom et al.), Adv = 2*AUC-1.
  - Bound factor: mean 5-NN distance u(x).  For uniform data the k-NN
    density estimate is constant (Prop. 2), so the geometric factor of
    Theorem 1 reduces to mean u(x) up to constants.

Output:
  results/synthetic_tightness.json   — all raw numbers
  paper/figures/tightness_plot.pdf   — log-log decay plot with fitted slopes

Seeds fixed; deterministic given the same sklearn version.
"""

import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIMS = [2, 4, 8]
NS = [250, 500, 1000, 2000, 4000]
SEEDS = [0, 1, 2, 3, 4]
LABEL_NOISE = 0.10
K = 5


def make_dataset(n, d, rng):
    """Uniform hypercube features, hyperplane labels, 10% label noise."""
    X = rng.uniform(size=(2 * n, d))  # half members, half non-members
    w = rng.normal(size=d)
    y = (X @ w > np.median(X @ w)).astype(int)
    flip = rng.uniform(size=2 * n) < LABEL_NOISE
    y = np.where(flip, 1 - y, y)
    return X[:n], y[:n], X[n:], y[n:]


def mean_uniqueness(X, k=K):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dists, _ = nn.kneighbors(X)
    return float(dists[:, k].mean())


def loss_threshold_advantage(X_mem, y_mem, X_non, y_non, seed):
    model = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=2000,
                          early_stopping=False, random_state=seed)
    model.fit(X_mem, y_mem)
    conf_mem = model.predict_proba(X_mem)[np.arange(len(y_mem)), y_mem]
    conf_non = model.predict_proba(X_non)[np.arange(len(y_non)), y_non]
    labels = np.r_[np.ones(len(conf_mem)), np.zeros(len(conf_non))]
    auc = roc_auc_score(labels, np.r_[conf_mem, conf_non])
    return max(0.0, 2 * auc - 1)  # advantage, clipped at 0


def loglog_slope(ns, vals):
    """OLS slope of log(vals) on log(ns)."""
    x, v = np.log(np.array(ns, float)), np.log(np.maximum(vals, 1e-6))
    return float(np.polyfit(x, v, 1)[0])


def main():
    results = {"dims": DIMS, "ns": NS, "seeds": SEEDS,
               "label_noise": LABEL_NOISE, "per_dim": {}}
    for d in DIMS:
        u_means, advs = [], []
        for n in NS:
            u_per_seed, adv_per_seed = [], []
            for seed in SEEDS:
                rng = np.random.default_rng(1000 * d + 10 * seed)
                X_mem, y_mem, X_non, y_non = make_dataset(n, d, rng)
                u_per_seed.append(mean_uniqueness(X_mem))
                adv_per_seed.append(
                    loss_threshold_advantage(X_mem, y_mem, X_non, y_non, seed))
            u_means.append(float(np.mean(u_per_seed)))
            advs.append(float(np.mean(adv_per_seed)))
            print(f"d={d} n={n}: mean_u={u_means[-1]:.4f} "
                  f"adv={advs[-1]:.4f}", flush=True)
        results["per_dim"][d] = {
            "mean_u": u_means,
            "advantage": advs,
            "slope_u": loglog_slope(NS, u_means),
            "slope_adv": loglog_slope(NS, advs),
            "predicted_slope": -1.0 / d,
        }
        print(f"d={d}: slope_u={results['per_dim'][d]['slope_u']:.3f} "
              f"slope_adv={results['per_dim'][d]['slope_adv']:.3f} "
              f"predicted={-1.0/d:.3f}", flush=True)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", "synthetic_tightness.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out}")
    return results


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    colors = {2: "tab:blue", 4: "tab:orange", 8: "tab:green"}
    ns = results["ns"]
    for d in results["dims"]:
        r = results["per_dim"][d]
        axes[0].loglog(ns, r["mean_u"], "o-", color=colors[d],
                       label=f"$d={d}$ (slope ${r['slope_u']:.2f}$, "
                             f"pred.\\ ${-1.0/d:.2f}$)")
        axes[1].loglog(ns, np.maximum(r["advantage"], 1e-3), "o-",
                       color=colors[d],
                       label=f"$d={d}$ (slope ${r['slope_adv']:.2f}$)")
    axes[0].set_xlabel("$n$")
    axes[0].set_ylabel(r"bound factor $\bar{u}$")
    axes[0].set_title("Geometric factor")
    axes[1].set_xlabel("$n$")
    axes[1].set_ylabel("measured MIA advantage")
    axes[1].set_title("Loss-threshold attack")
    for ax in axes:
        ax.legend(fontsize=6.5)
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    figpath = os.path.join(ROOT, "paper", "figures", "tightness_plot.pdf")
    fig.savefig(figpath, bbox_inches="tight")
    print(f"saved {figpath}")


if __name__ == "__main__":
    res = main()
    if "--no-plot" not in sys.argv:
        plot(res)
