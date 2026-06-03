"""
DPRI Command-Line Tool (Phase 4, Task 4.3)

Computes the Dataset Privacy Risk Index for any CSV or .npz dataset.

Usage:
    python -m src.dpri.cli --input data.csv --label income
    python -m src.dpri.cli --input data.npz
    python -m src.dpri.cli --input data.csv --label income --output report.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.dpri.features import compute_dpri_features

# Risk thresholds calibrated on experimental datasets (update after Phase 3)
RISK_THRESHOLDS = {
    "Low":    (0.0,  0.35),
    "Medium": (0.35, 0.65),
    "High":   (0.65, 1.0),
}


def load_csv(path: Path, label_col: str):
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Columns: {list(df.columns)}")
    y_raw = df[label_col]
    X_raw = df.drop(columns=[label_col])

    # encode categoricals
    for col in X_raw.select_dtypes("object").columns:
        X_raw[col] = LabelEncoder().fit_transform(X_raw[col].astype(str))

    X = X_raw.fillna(X_raw.median()).values.astype(np.float32)
    y = LabelEncoder().fit_transform(y_raw.astype(str)).astype(np.int32)
    X = StandardScaler().fit_transform(X)
    return X, y


def load_npz(path: Path):
    d = np.load(path)
    return d["X"].astype(np.float32), d["y"].astype(np.int32)


def dpri_to_scalar(feats: dict) -> float:
    """
    Combine DPRI features into a single risk score in [0, 1].
    Weights are preliminary; will be replaced by regression coefficients
    after Phase 3 Task 3.4 completes.
    """
    # Normalize each feature heuristically; signs chosen so high = more risk
    score = (
        0.30 * min(feats["uniqueness_mean"] / 2.0, 1.0) +   # high uniqueness → risk
        0.20 * max(0, 1.0 - feats["density_mean"] / 5.0) +  # low density → risk
        0.20 * feats["outlier_mean"] +                        # high outlier → risk
        0.15 * min(feats["entropy"] / 10.0, 1.0) +           # high entropy → risk
        0.15 * max(0, feats["cluster_sep"])                   # high separation → risk
    )
    return float(np.clip(score, 0.0, 1.0))


def risk_category(score: float) -> str:
    for cat, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= score < hi:
            return cat
    return "High"


def print_report(feats: dict, scalar: float, category: str, dataset_name: str):
    width = 54
    bar_len = int(scalar * 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)

    print("\n" + "="*width)
    print(f"  DPRI Report: {dataset_name}")
    print("="*width)
    print(f"  Overall Risk Score : {scalar:.3f}  [{bar}]")
    print(f"  Risk Category      : {category}")
    print("-"*width)
    print("  Feature Breakdown:")
    feature_labels = {
        "uniqueness_mean": "Sample Uniqueness (mean)",
        "uniqueness_p90":  "Sample Uniqueness (p90)",
        "density_mean":    "Local Density (mean)",
        "density_p90":     "Local Density (p90)",
        "outlier_mean":    "Outlier Score (mean)",
        "outlier_p90":     "Outlier Score (p90)",
        "entropy":         "Feature Entropy",
        "cluster_sep":     "Cluster Separation",
    }
    for key, label in feature_labels.items():
        if key in feats:
            print(f"    {label:<30} {feats[key]:.4f}")
    print("-"*width)
    if category == "High":
        print("  Recommendation: Apply DP-SGD with epsilon <= 1.0 before")
        print("  training. Consider data minimization or anonymization.")
    elif category == "Medium":
        print("  Recommendation: Consider DP-SGD with epsilon <= 5.0.")
        print("  Evaluate access controls before model release.")
    else:
        print("  Recommendation: Standard privacy practices are likely")
        print("  sufficient. Recheck if dataset composition changes.")
    print("="*width + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute DPRI for a dataset before model training."
    )
    parser.add_argument("--input",  required=True, help="Path to .csv or .npz file")
    parser.add_argument("--label",  default=None,  help="Label column name (CSV only)")
    parser.add_argument("--output", default=None,  help="Optional JSON output path")
    parser.add_argument("--k",      type=int, default=5, help="k for k-NN features")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    if path.suffix == ".csv":
        if not args.label:
            print("ERROR: --label is required for CSV files.", file=sys.stderr)
            sys.exit(1)
        X, y = load_csv(path, args.label)
    elif path.suffix == ".npz":
        X, y = load_npz(path)
    else:
        print(f"ERROR: Unsupported format '{path.suffix}'. Use .csv or .npz", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded: {len(X)} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    feats = compute_dpri_features(X, y, k=args.k)
    scalar = dpri_to_scalar(feats)
    category = risk_category(scalar)

    print_report(feats, scalar, category, path.name)

    if args.output:
        out = {"dataset": path.name, "dpri_score": scalar,
               "risk_category": category, "features": feats}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
