"""
Rebuttal (#1613), Reviewer C (Corollary 1): subsampling sweep.

For a few large datasets, retain a fraction f of the records and measure
  * mean 5-NN uniqueness u and mean floored density rho (DPRI inputs), and
  * loss-threshold attack AUC for the MLP and XGBoost families of the grid,
as f shrinks. The proposition predicts E[u] ∝ n^{-1/d}; the sweep reports the
fitted log-log slope of u against n together with the measured AUC trend.

    python experiments/rebuttal_subsample_sweep.py \
        [--datasets adult,covtype,mnist,purchase100] [--fractions 1,0.5,0.25,0.125,0.0625] \
        [--seeds 0,1,2] [--models mlp,xgboost]
    -> results/rebuttal/subsample_sweep.json, results/rebuttal/figures/subsample_sweep.pdf
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.classifiers import get_model  # noqa: E402
from src.mia.attacks import attack_loss_threshold  # noqa: E402
from src.dpri.features import compute_density  # noqa: E402

OUT = Path("results/rebuttal")
FIG = OUT / "figures"
K = 5


def geometry(X):
    nn = NearestNeighbors(n_neighbors=K + 1, n_jobs=-1).fit(X)
    dists, _ = nn.kneighbors(X)
    u = dists[:, K]
    rho = compute_density(X, k=K)
    return float(u.mean()), float(rho.mean())


def safe_split(X, y, seed):
    try:
        return train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    except ValueError:           # a class with a single sample after subsampling
        return train_test_split(X, y, test_size=0.5, random_state=seed)


def loglog_slope(ns, vals):
    x, v = np.log(np.asarray(ns, float)), np.log(np.maximum(np.asarray(vals, float), 1e-9))
    return float(np.polyfit(x, v, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="adult,covtype,mnist,purchase100")
    ap.add_argument("--fractions", default="1,0.5,0.25,0.125,0.0625")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--models", default="mlp,xgboost")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--min_n", type=int, default=400)
    args = ap.parse_args()

    datasets = [s for s in args.datasets.split(",") if s]
    fractions = [float(f) for f in args.fractions.split(",") if f]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    models = [m for m in args.models.split(",") if m]
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    results = {"fractions": fractions, "seeds": seeds, "models": models, "per_dataset": {}}
    for ds in datasets:
        path = Path(args.data_dir) / f"{ds}.npz"
        if not path.exists():
            print(f"SKIP {ds}: {path} missing")
            continue
        d = np.load(path)
        X, y = d["X"], d["y"]
        n_full, dim = X.shape
        rows = []
        for f in fractions:
            n_sub = max(args.min_n, int(round(n_full * f)))
            n_sub = min(n_sub, n_full)
            for seed in seeds:
                rng = np.random.default_rng(1000 * seed + int(1000 * f))
                idx = rng.choice(n_full, n_sub, replace=False)
                Xs, ys = X[idx], y[idx]
                t0 = time.time()
                u_mean, rho_mean = geometry(Xs)
                Xtr, Xte, ytr, yte = safe_split(Xs, ys, seed)
                n_classes = len(np.unique(ys))
                aucs = {}
                for m in models:
                    try:
                        model = get_model(m, n_classes=n_classes, seed=seed)
                        model.fit(Xtr, ytr)
                        _, _, auc = attack_loss_threshold(model, Xtr, ytr, Xte, yte)
                        aucs[m] = float(auc)
                    except Exception as e:      # e.g. a class absent from the member half
                        aucs[m] = None
                        print(f"    {m} failed at f={f} seed={seed}: {type(e).__name__}: {e}")
                rows.append({"fraction": f, "seed": seed, "n": int(n_sub),
                             "u_mean": u_mean, "rho_mean": rho_mean, "auc": aucs})
                print(f"  {ds} f={f:<6} seed={seed} n={n_sub:>6} u={u_mean:.4f} rho={rho_mean:.3f} "
                      f"auc={ {k: (round(v,3) if v is not None else None) for k,v in aucs.items()} } "
                      f"({time.time()-t0:.0f}s)", flush=True)
        # aggregate per fraction
        agg = []
        for f in fractions:
            rr = [r for r in rows if r["fraction"] == f]
            if not rr:
                continue
            entry = {"fraction": f, "n": int(np.mean([r["n"] for r in rr])),
                     "u_mean": float(np.mean([r["u_mean"] for r in rr])),
                     "rho_mean": float(np.mean([r["rho_mean"] for r in rr]))}
            for m in models:
                vals = [r["auc"][m] for r in rr if r["auc"].get(m) is not None]
                entry[f"auc_{m}"] = float(np.mean(vals)) if vals else None
            agg.append(entry)
        ns = [a["n"] for a in agg]
        results["per_dataset"][ds] = {
            "n_full": int(n_full), "d": int(dim), "rows": rows, "aggregate": agg,
            "slope_u_vs_n": loglog_slope(ns, [a["u_mean"] for a in agg]) if len(agg) > 1 else None,
            "predicted_slope_minus_1_over_d": -1.0 / dim,
        }
        with open(OUT / "subsample_sweep.json", "w") as fh:
            json.dump(results, fh, indent=2)

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        for ds, r in results["per_dataset"].items():
            ns = [a["n"] for a in r["aggregate"]]
            axes[0].loglog(ns, [a["u_mean"] for a in r["aggregate"]], "o-",
                           label=f"{ds} (d={r['d']}, slope {r['slope_u_vs_n']:.2f})")
            for m in models:
                vals = [a.get(f"auc_{m}") for a in r["aggregate"]]
                if any(v is not None for v in vals):
                    axes[1].semilogx(ns, [v if v is not None else np.nan for v in vals],
                                     "o-" if m == "mlp" else "s--", label=f"{ds} {m}")
        axes[0].set_xlabel("n (retained samples)"); axes[0].set_ylabel("mean 5-NN uniqueness")
        axes[1].set_xlabel("n (retained samples)"); axes[1].set_ylabel("loss-threshold AUC")
        for ax in axes:
            ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(FIG / "subsample_sweep.pdf", bbox_inches="tight")
        fig.savefig(FIG / "subsample_sweep.png", dpi=160, bbox_inches="tight")
        print(f"saved {FIG / 'subsample_sweep.pdf'}")
    except Exception as e:
        print(f"plot skipped: {e}")
    print(f"saved {OUT / 'subsample_sweep.json'}")


if __name__ == "__main__":
    main()
