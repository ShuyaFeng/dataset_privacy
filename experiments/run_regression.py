"""
Phase 3 Task 3.4 — DPRI → Risk(D) Regression

Requires:
  results/ground_truth_risk.csv   (from check_ground_truth_variance.py)
  results/dpri/dpri_features.csv  (from run_dpri.py)

Output:
  results/regression/regression_results.json
  results/regression/regression_plot.pdf

Usage:
    python experiments/run_regression.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

FEATURE_COLS = [
    "uniqueness_mean", "uniqueness_p90",
    "density_mean",    "density_p90",
    "outlier_mean",    "outlier_p90",
    "entropy",
    "cluster_sep",
]

OUT_DIR = Path("results/regression")


def load_data():
    risk_path  = Path("results/ground_truth_risk.csv")
    dpri_path  = Path("results/dpri/dpri_features.csv")

    if not risk_path.exists():
        raise FileNotFoundError(f"Missing {risk_path} — run check_ground_truth_variance.py first")
    if not dpri_path.exists():
        raise FileNotFoundError(f"Missing {dpri_path} — run run_dpri.py first")

    risk = pd.read_csv(risk_path).set_index("dataset")
    dpri = pd.read_csv(dpri_path, index_col=0)

    df = dpri.join(risk, how="inner")
    print(f"Datasets with both DPRI and Risk(D): {list(df.index)}")
    return df


def run_loo_cv(X: np.ndarray, y: np.ndarray, model_name: str):
    """Leave-one-out cross-validation."""
    if model_name == "linear":
        model_cls = lambda: Ridge(alpha=0.1)
    elif model_name == "rf":
        model_cls = lambda: RandomForestRegressor(n_estimators=500, random_state=42)
    else:
        raise ValueError(model_name)

    loo = LeaveOneOut()
    y_true, y_pred = [], []

    for train_idx, test_idx in loo.split(X):
        m = model_cls()
        m.fit(X[train_idx], y[train_idx])
        y_true.append(y[test_idx[0]])
        y_pred.append(m.predict(X[test_idx])[0])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    r2 = r2_score(y_true, y_pred)
    spear, _ = spearmanr(y_true, y_pred)
    return r2, spear, y_true, y_pred


def ablation(X: np.ndarray, y: np.ndarray, feature_names: list):
    """Drop one feature at a time, measure R² drop."""
    baseline_r2, _, _, _ = run_loo_cv(X, y, "linear")
    results = {}
    for i, fname in enumerate(feature_names):
        mask = [j for j in range(len(feature_names)) if j != i]
        r2_drop, _, _, _ = run_loo_cv(X[:, mask], y, "linear")
        results[fname] = {
            "r2_without": round(r2_drop, 4),
            "r2_drop":    round(baseline_r2 - r2_drop, 4),
        }
    return baseline_r2, results


def make_plot(y_true, y_pred_linear, y_pred_rf, dataset_names, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, y_pred, title in zip(
        axes,
        [y_pred_linear, y_pred_rf],
        ["Linear Regression (LOO-CV)", "Random Forest (LOO-CV)"],
    ):
        ax.scatter(y_true, y_pred, zorder=3, s=80, edgecolors="k", linewidths=0.5)
        for i, name in enumerate(dataset_names):
            ax.annotate(name, (y_true[i], y_pred[i]), fontsize=7,
                        xytext=(4, 2), textcoords="offset points")
        lo = min(y_true.min(), y_pred.min()) - 0.02
        hi = max(y_true.max(), y_pred.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y=x")
        ax.set_xlabel("Observed Risk(D) (AUC)")
        ax.set_ylabel("Predicted Risk(D)")
        ax.set_title(title)
        r2 = r2_score(y_true, y_pred)
        spear, _ = spearmanr(y_true, y_pred)
        ax.text(0.05, 0.92, f"$R^2$={r2:.3f}  $\\rho$={spear:.3f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Plot saved: {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X = df[FEATURE_COLS].values.astype(np.float64)
    y = df["Risk_D"].values.astype(np.float64)
    dataset_names = list(df.index)

    print("\n" + "="*60)
    print("Leave-One-Out Cross-Validation")
    print("="*60)

    r2_lin, sp_lin, yt, yp_lin = run_loo_cv(X, y, "linear")
    r2_rf,  sp_rf,  _,  yp_rf  = run_loo_cv(X, y, "rf")

    print(f"  Linear:  R²={r2_lin:.4f}  Spearman={sp_lin:.4f}")
    print(f"  RF:      R²={r2_rf:.4f}  Spearman={sp_rf:.4f}")

    # ── Variance decomposition (Finding 1 support) ───────────────────────
    # Fit full model once to get feature importances
    full_rf = RandomForestRegressor(n_estimators=500, random_state=42)
    full_rf.fit(X, y)
    importances = dict(zip(FEATURE_COLS, full_rf.feature_importances_))

    print("\n" + "="*60)
    print("RF Feature Importances (proxy for variance explained)")
    print("="*60)
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat:<25} {imp:.4f}")

    # ── Ablation ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Ablation (Linear, LOO-CV)")
    print("="*60)
    baseline_r2, abl = ablation(X, y, FEATURE_COLS)
    print(f"  Baseline R²: {baseline_r2:.4f}")
    for feat, res in sorted(abl.items(), key=lambda x: -x[1]["r2_drop"]):
        flag = "  <-- key" if res["r2_drop"] > 0.03 else ""
        print(f"  drop {feat:<25}  R²={res['r2_without']:.4f}  Δ={res['r2_drop']:+.4f}{flag}")

    # ── Ground-truth variance check (Finding 1) ──────────────────────────
    mia_dir = Path("results/mia_grid")
    if mia_dir.exists() and any(mia_dir.glob("*.json")):
        import json as _json
        records = [_json.load(open(p)) for p in mia_dir.glob("*.json")]
        mia_df = pd.DataFrame(records)

        var_by_dataset = mia_df.groupby("dataset")["auc"].var().mean()
        var_by_model   = mia_df.groupby("model")["auc"].var().mean()
        var_by_attack  = mia_df.groupby("attack")["auc"].var().mean()

        print("\n" + "="*60)
        print("Finding 1: Variance Decomposition")
        print("="*60)
        total = var_by_dataset + var_by_model + var_by_attack
        print(f"  Variance from dataset:  {var_by_dataset/total*100:.1f}%")
        print(f"  Variance from model:    {var_by_model/total*100:.1f}%")
        print(f"  Variance from attack:   {var_by_attack/total*100:.1f}%")

    # ── Save results ─────────────────────────────────────────────────────
    results = {
        "linear": {"r2_loo": round(r2_lin, 4), "spearman_loo": round(sp_lin, 4)},
        "rf":     {"r2_loo": round(r2_rf,  4), "spearman_loo": round(sp_rf,  4)},
        "ablation": abl,
        "feature_importances": {k: round(v, 4) for k, v in importances.items()},
        "datasets": dataset_names,
        "y_true": list(yt.round(4)),
        "y_pred_linear": list(yp_lin.round(4)),
        "y_pred_rf":     list(yp_rf.round(4)),
    }
    out_json = OUT_DIR / "regression_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")

    make_plot(yt, yp_lin, yp_rf, dataset_names, OUT_DIR / "regression_plot.pdf")

    # ── Print success/failure verdict ────────────────────────────────────
    print("\n" + "="*60)
    print("VERDICT (per plan.md Task 3.4 success criteria)")
    print("="*60)
    if r2_lin >= 0.90:
        print(f"  Linear R²={r2_lin:.4f} ≥ 0.90 → EXCELLENT")
    elif r2_lin >= 0.85:
        print(f"  Linear R²={r2_lin:.4f} ≥ 0.85 → STRONG")
    elif r2_lin >= 0.75:
        print(f"  Linear R²={r2_lin:.4f} ≥ 0.75 → ACCEPTABLE")
    else:
        print(f"  Linear R²={r2_lin:.4f} < 0.75 → REVIEW NEEDED (see plan.md mitigation)")

    if sp_lin >= 0.8:
        print(f"  Spearman ρ={sp_lin:.4f} ≥ 0.80 → PASS")
    else:
        print(f"  Spearman ρ={sp_lin:.4f} < 0.80 → REVIEW NEEDED")


if __name__ == "__main__":
    main()
