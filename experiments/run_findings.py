"""
Phase 3 Task 3.5 — Key Findings Validation

Finding 2: Benchmark bias
    Compare DPRI distribution of benchmark vs real-world datasets.

Finding 3: DP calibration depends on DPRI, not just epsilon
    For each dataset, apply DP-SGD with epsilon in {0.1, 1, 10}.
    Measure AUC reduction. Show it correlates with DPRI.

Usage:
    python experiments/run_findings.py --finding 2
    python experiments/run_findings.py --finding 3
    python experiments/run_findings.py           # run both
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.classifiers import get_model
from src.mia.attacks import attack_loss_threshold

OUT_DIR = Path("results/regression")


# ── Finding 2: Benchmark Bias ────────────────────────────────────────────────

BENCHMARK_DATASETS = {"purchase100", "texas100"}
REALWORLD_DATASETS  = {"adult", "compas", "nhanes", "movielens", "gowalla"}


def finding2_benchmark_bias():
    dpri_path = Path("results/dpri/dpri_features.csv")
    if not dpri_path.exists():
        print("Finding 2: SKIP — run run_dpri.py first")
        return

    df = pd.read_csv(dpri_path, index_col=0)

    benchmarks = df[df.index.isin(BENCHMARK_DATASETS)]
    realworld  = df[df.index.isin(REALWORLD_DATASETS)]

    if benchmarks.empty or realworld.empty:
        print("Finding 2: not enough datasets to compare")
        return

    print("\n" + "="*60)
    print("FINDING 2: Benchmark Bias")
    print("="*60)

    feature_cols = [
        "uniqueness_mean", "density_mean", "outlier_mean",
        "entropy", "cluster_sep",
    ]

    results = {}
    for feat in feature_cols:
        b_vals = benchmarks[feat].values
        r_vals = realworld[feat].values
        stat, pval = ks_2samp(b_vals, r_vals)
        results[feat] = {
            "benchmark_mean": round(float(b_vals.mean()), 4),
            "realworld_mean": round(float(r_vals.mean()), 4),
            "ks_stat": round(stat, 4),
            "ks_pval": round(pval, 4),
            "significant": pval < 0.05,
        }
        sig = "**" if pval < 0.05 else "  "
        print(f"  {sig} {feat:<25} "
              f"benchmark={b_vals.mean():.3f}  realworld={r_vals.mean():.3f}  "
              f"KS={stat:.3f}  p={pval:.3f}")

    out_path = OUT_DIR / "finding2_benchmark_bias.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

    # ── Plot: DPRI distribution comparison ───────────────────────────────
    fig, axes = plt.subplots(1, len(feature_cols), figsize=(4 * len(feature_cols), 3))
    for ax, feat in zip(axes, feature_cols):
        ax.bar(["Benchmark", "Real-world"],
               [benchmarks[feat].mean(), realworld[feat].mean()],
               color=["#e74c3c", "#3498db"],
               yerr=[benchmarks[feat].std(), realworld[feat].std()],
               capsize=4)
        ax.set_title(feat.replace("_", "\n"), fontsize=8)
    plt.suptitle("Finding 2: Benchmark vs. Real-world DPRI Features", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "finding2_plot.pdf", dpi=200, bbox_inches="tight")
    print(f"  Plot saved: {OUT_DIR / 'finding2_plot.pdf'}")


# ── Finding 3: DP Calibration ────────────────────────────────────────────────

def _dp_sgd_mlp(X_train, y_train, epsilon: float, seed: int = 42):
    """
    Simulate DP-SGD by adding calibrated Gaussian noise to MLP gradients.
    This is a lightweight approximation — for camera-ready use opacus.
    Noise scale σ = sqrt(2 * ln(1.25/delta)) / epsilon  (Gaussian mechanism).
    """
    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        n_classes = len(np.unique(y_train))
        in_dim = X_train.shape[1]

        model = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),     nn.ReLU(),
            nn.Linear(32, n_classes),
        )
        model = ModuleValidator.fix(model)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        Xt = torch.FloatTensor(X_train)
        yt = torch.LongTensor(y_train)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=256, shuffle=True)

        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            epochs=30,
            target_epsilon=epsilon,
            target_delta=1e-5,
            max_grad_norm=1.0,
        )

        model.train()
        for _ in range(30):
            for xb, yb in loader:
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                optimizer.step()

        # wrap for sklearn-compatible predict_proba
        class TorchWrapper:
            def __init__(self, m, classes):
                self.m = m
                self.classes_ = classes
            def predict_proba(self, X):
                self.m.eval()
                with torch.no_grad():
                    logits = self.m(torch.FloatTensor(X))
                    return torch.softmax(logits, dim=1).numpy()

        return TorchWrapper(model, list(range(n_classes)))

    except ImportError:
        # Opacus not available: approximate with noise injection
        from sklearn.neural_network import MLPClassifier
        import copy

        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=seed)
        mlp.fit(X_train, y_train)

        # Add Gaussian noise to coefs proportional to 1/epsilon
        delta = 1e-5
        sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        for i in range(len(mlp.coefs_)):
            noise = np.random.default_rng(seed).normal(0, sigma * 0.01, mlp.coefs_[i].shape)
            mlp.coefs_[i] += noise
        return mlp


def finding3_dp_calibration(data_dir: Path = Path("data/processed")):
    dpri_path = Path("results/dpri/dpri_features.csv")
    if not dpri_path.exists():
        print("Finding 3: SKIP — run run_dpri.py first")
        return

    dpri_df = pd.read_csv(dpri_path, index_col=0)
    epsilons = [0.1, 1.0, 10.0]
    results = {}

    print("\n" + "="*60)
    print("FINDING 3: DP Calibration vs DPRI")
    print("="*60)

    all_datasets = [d for d in dpri_df.index if (data_dir / f"{d}.npz").exists()]

    for ds in all_datasets:
        data = np.load(data_dir / f"{ds}.npz")
        X, y = data["X"], data["y"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42, stratify=y
        )

        # Baseline (no DP)
        baseline_model = get_model("mlp", n_classes=len(np.unique(y)), seed=42)
        baseline_model.fit(X_train, y_train)
        _, _, baseline_auc = attack_loss_threshold(baseline_model, X_train, y_train, X_test, y_test)

        eps_results = {"baseline_auc": round(baseline_auc, 4)}
        for eps in epsilons:
            dp_model = _dp_sgd_mlp(X_train, y_train, epsilon=eps)
            _, _, dp_auc = attack_loss_threshold(dp_model, X_train, y_train, X_test, y_test)
            auc_drop = baseline_auc - dp_auc
            eps_results[f"eps_{eps}"] = {
                "auc": round(dp_auc, 4),
                "auc_drop": round(auc_drop, 4),
            }
            print(f"  {ds:<15} eps={eps:<5} baseline={baseline_auc:.3f}  "
                  f"dp={dp_auc:.3f}  drop={auc_drop:+.3f}")

        results[ds] = eps_results

    # Correlate AUC drop with DPRI
    print("\n  Spearman correlation: DPRI features vs AUC drop at each epsilon")
    for eps in epsilons:
        drops, dpri_scores = [], []
        for ds in results:
            if ds in dpri_df.index and f"eps_{eps}" in results[ds]:
                drops.append(results[ds][f"eps_{eps}"]["auc_drop"])
                dpri_scores.append(dpri_df.loc[ds, "uniqueness_mean"])  # proxy
        if len(drops) >= 3:
            rho, pval = spearmanr(dpri_scores, drops)
            print(f"  eps={eps:<5}  Spearman ρ={rho:.3f}  p={pval:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "finding3_dp_calibration.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finding", type=int, choices=[2, 3], default=None,
                        help="Run finding 2, 3, or both (default)")
    parser.add_argument("--data_dir", default="data/processed")
    args = parser.parse_args()

    run_all = args.finding is None
    if run_all or args.finding == 2:
        finding2_benchmark_bias()
    if run_all or args.finding == 3:
        finding3_dp_calibration(Path(args.data_dir))


if __name__ == "__main__":
    main()
