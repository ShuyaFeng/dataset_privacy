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
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

# DPRI's five concepts, one statistic each. The p90 variants are dropped for
# the regression: they correlate 0.98-0.99 with their means, adding
# collinearity without information at n=7. This does NOT change DPRI's
# definition; these are exactly the five concepts named in the paper
# (uniqueness, density, outlier, entropy, class separability).
FEATURE_COLS = [
    "uniqueness_mean",
    "density_mean",
    "outlier_mean",
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
    # alpha is high because n is tiny (7 datasets) and features are collinear
    # (uniqueness and density are near-reciprocals); strong shrinkage is needed.
    if model_name == "linear":
        model_cls = lambda: Ridge(alpha=1.0)
    elif model_name == "rf":
        model_cls = lambda: RandomForestRegressor(n_estimators=500, random_state=42)
    else:
        raise ValueError(model_name)

    loo = LeaveOneOut()
    y_true, y_pred = [], []

    for train_idx, test_idx in loo.split(X):
        # Rank-transform IN-FOLD (map test through the train ECDF), then
        # standardize. Geometric features are heavily skewed (density spans
        # 0.03..35); on raw values the regression is dominated by outliers and
        # LOO collapses (density alone gives Spearman 0.05). Rank-transform
        # fixes this and is natural since the target metric is rank correlation.
        Xtr_raw, Xte_raw = X[train_idx], X[test_idx]
        Xtr = np.empty_like(Xtr_raw, dtype=float)
        Xte = np.empty_like(Xte_raw, dtype=float)
        for j in range(Xtr_raw.shape[1]):
            col = Xtr_raw[:, j]
            Xtr[:, j] = rankdata(col)
            Xte[:, j] = [(col < v).sum() + 0.5 * (col == v).sum() for v in Xte_raw[:, j]]
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
        m = model_cls()
        m.fit(Xtr, y[train_idx])
        y_true.append(y[test_idx[0]])
        y_pred.append(m.predict(Xte)[0])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    r2 = r2_score(y_true, y_pred)
    spear, _ = spearmanr(y_true, y_pred)
    return r2, spear, y_true, y_pred


def single_factor_analysis(df: pd.DataFrame, feature_names: list):
    """Each feature's standalone Spearman rank correlation with Risk(D).

    This replaces a drop-one ablation, which is not meaningful here: the
    linear baseline R^2 is negative (deltas off it are noise) and the
    features are collinear (dropping one lets a correlated feature absorb its
    signal). Standalone rank correlation is robust to both and directly shows
    which geometric factor carries the predictive signal, with a p-value that
    honestly reflects the small sample size.
    """
    y = df["Risk_D"].values
    results = {}
    for f in feature_names:
        rho, p = spearmanr(df[f].values, y)
        results[f] = {"spearman": round(float(rho), 4), "p_value": round(float(p), 4)}
    return results


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

    df = load_data().copy()
    # Dimensionality is an independent risk signal: higher-dimensional data is
    # sparser, so samples are more memorizable. Add log(n_features) alongside
    # the five geometric features.
    if "n_features" in df.columns:
        df["log_nfeatures"] = np.log(df["n_features"].clip(lower=1))
        feat_list = FEATURE_COLS + ["log_nfeatures"]
    else:
        feat_list = list(FEATURE_COLS)
    X = df[feat_list].values.astype(np.float64)
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
    importances = dict(zip(feat_list, full_rf.feature_importances_))

    print("\n" + "="*60)
    print("RF Feature Importances (proxy for variance explained)")
    print("="*60)
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat:<25} {imp:.4f}")

    # ── Single-factor analysis (replaces drop-one ablation) ──────────────
    print("\n" + "="*60)
    print("Single-factor ranking power (Spearman vs Risk(D))")
    print("="*60)
    sf = single_factor_analysis(df, feat_list)
    for feat, res in sorted(sf.items(), key=lambda x: -abs(x[1]["spearman"])):
        sig = "significant" if res["p_value"] < 0.05 else "(n.s.)"
        print(f"  {feat:<18} rho={res['spearman']:+.3f}  p={res['p_value']:.3f}  {sig}")

    # ── Ground-truth variance check (Finding 1) ──────────────────────────
    mia_dir = Path("results/mia_grid")
    if mia_dir.exists() and any(mia_dir.glob("*.json")):
        import json as _json
        records = [_json.load(open(p)) for p in mia_dir.glob("*.json")]
        mia_df = pd.DataFrame(records)

        # Factor influence = spread (std) of that factor's GROUP MEANS.
        # Large spread means changing the factor moves AUC a lot, i.e. it
        # explains more of the variation. (The previous version used
        # groupby.var().mean(), which is WITHIN-group variance: the opposite
        # signal, and it was mislabeled as "variance from X".)
        dataset_spread = mia_df.groupby("dataset")["auc"].mean().std()
        model_spread   = mia_df.groupby("model")["auc"].mean().std()
        attack_spread  = mia_df.groupby("attack")["auc"].mean().std()

        print("\n" + "="*60)
        print("Finding 1: Factor influence (std of group-mean AUC)")
        print("="*60)
        total = dataset_spread + model_spread + attack_spread
        print(f"  Dataset: std={dataset_spread:.4f}  ({dataset_spread/total*100:.1f}%)")
        print(f"  Model:   std={model_spread:.4f}  ({model_spread/total*100:.1f}%)")
        print(f"  Attack:  std={attack_spread:.4f}  ({attack_spread/total*100:.1f}%)")

    # ── Save results ─────────────────────────────────────────────────────
    results = {
        "linear": {"r2_loo": round(r2_lin, 4), "spearman_loo": round(sp_lin, 4)},
        "rf":     {"r2_loo": round(r2_rf,  4), "spearman_loo": round(sp_rf,  4)},
        "single_factor": sf,
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

    # ── Verdict ──────────────────────────────────────────────────────────
    # Primary metric is rank correlation (Spearman). Features are rank-
    # transformed and heavily skewed, so absolute-value R^2 is not the right
    # lens; what a pre-training index needs is to rank datasets by risk.
    print("\n" + "="*60)
    print("VERDICT (primary metric: rank correlation, n=31)")
    print("="*60)
    print(f"  Linear (rank+ridge) Spearman  rho={sp_lin:.4f}")
    print(f"  RF                  Spearman  rho={sp_rf:.4f}")
    print(f"  (R^2: Linear={r2_lin:.3f}, RF={r2_rf:.3f})")
    best_rho = max(sp_rf, sp_lin)
    if best_rho >= 0.75:
        print(f"  -> STRONG: DPRI ranks dataset risk well (rho={best_rho:.3f})")
    elif best_rho >= 0.55:
        print(f"  -> MODERATE: DPRI is a significant moderate predictor (rho={best_rho:.3f})")
    elif best_rho >= 0.40:
        print(f"  -> WEAK-MODERATE (rho={best_rho:.3f}); signal present but limited")
    else:
        print(f"  -> WEAK (rho={best_rho:.3f}); revisit features or ground truth")
    print("  CAVEAT: feature choice and transform were selected during analysis;")
    print("  a nested-CV estimate is the honest unbiased number (likely a few")
    print("  points lower). Report it as such in the paper.")


if __name__ == "__main__":
    main()
