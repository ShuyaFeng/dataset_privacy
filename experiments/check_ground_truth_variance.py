"""
Phase 1 Task 1.1 — Ground Truth Variance Check
===============================================
After the MIA grid finishes, run this to check whether AUC is stable
across attacks/models within each dataset.

If variance < 0.05 within each dataset → ground truth is reliable.
If variance >= 0.05 → follow the mitigation plan in plan.md Task 1.1.

Usage:
    python experiments/check_ground_truth_variance.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("results/mia_grid")


def load_results():
    rows = []
    for p in RESULTS_DIR.glob("*.json"):
        with open(p) as f:
            rows.append(json.load(f))
    return pd.DataFrame(rows)


def main():
    df = load_results()
    if df.empty:
        print("No results found. Run slurm/mia_grid_array.sh first.")
        return

    print(f"Loaded {len(df)} results.\n")
    print("=" * 60)
    print("AUC by dataset (mean ± std across 9 attack/model configs)")
    print("=" * 60)

    summary = df.groupby("dataset")["auc"].agg(["mean","std","min","max","count"])
    summary.columns = ["mean_auc", "std_auc", "min_auc", "max_auc", "n_configs"]
    summary = summary.sort_values("mean_auc", ascending=False)
    print(summary.round(4).to_string())

    print("\n" + "=" * 60)
    print("Variance check (std < 0.05 = PASS, >= 0.05 = REVIEW)")
    print("=" * 60)
    for ds, row in summary.iterrows():
        status = "PASS" if row["std_auc"] < 0.05 else "REVIEW"
        print(f"  {ds:<15} std={row['std_auc']:.4f}  [{status}]")

    print("\n" + "=" * 60)
    print("AUC by attack (aggregated across datasets and models)")
    print("=" * 60)
    print(df.groupby("attack")["auc"].agg(["mean","std"]).round(4).to_string())

    print("\n" + "=" * 60)
    print("AUC by model (aggregated across datasets and attacks)")
    print("=" * 60)
    print(df.groupby("model")["auc"].agg(["mean","std"]).round(4).to_string())

    # Ground truth options:
    #   mean = typical risk across the attack/model grid. Primary choice: it
    #          preserves dataset separation and is robust to the pathological
    #          over-fitting of unregularized RF.
    #   max  = worst-case adversary (upper envelope). Kept for comparison; it
    #          collapses onto RF for every dataset, erasing benchmark contrast.
    risk_mean = df.groupby("dataset")["auc"].mean().rename("Risk_D")
    risk_max  = df.groupby("dataset")["auc"].max().rename("Risk_D")

    print("\n" + "=" * 60)
    print("RISK(D): mean (primary) vs max (comparison)")
    print("=" * 60)
    cmp = pd.DataFrame({"mean": risk_mean, "max": risk_max})
    print(cmp.sort_values("mean", ascending=False).round(4).to_string())

    risk_mean.reset_index().to_csv("results/ground_truth_risk.csv", index=False)
    risk_max.reset_index().to_csv("results/ground_truth_risk_max.csv", index=False)
    print("\nSaved mean -> results/ground_truth_risk.csv (primary, used by regression)")
    print("Saved max  -> results/ground_truth_risk_max.csv (comparison)")


if __name__ == "__main__":
    main()
